#!/usr/bin/env python3
"""
Единая библиотека для логирования с поддержкой локального файла и Sentry.
Включает механизм повторных попыток отправки с очередью.
"""

import os
import sys
import atexit
import signal
import inspect
import logging
import threading
import queue
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration


def sentry_get_config():
    """Получение конфигурации Sentry"""
    environment = os.getenv("SENTRY_ENVIRONMENT", "development")
    is_production = environment == "production"

    config = {
        "dsn": os.getenv("SENTRY_DSN"),
        "environment": environment,
        "release": os.getenv("SENTRY_RELEASE", "dev"),
        "send_default_pii": is_production,
    }

    if is_production:
        config.update(
            {
                "traces_sample_rate": float(
                    os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.2")
                ),
                "profiles_sample_rate": float(
                    os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.1")
                ),
                "sample_rate": float(os.getenv("SENTRY_SAMPLE_RATE", "1.0")),
                "integrations": [
                    LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
                ],
            }
        )
    else:
        config.update(
            {
                "traces_sample_rate": float(
                    os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")
                ),
                "profiles_sample_rate": float(
                    os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0")
                ),
                "debug": os.getenv("SENTRY_DEBUG", "false").lower() == "true",
                "ignore_errors": ["KeyboardInterrupt", "SystemExit"],
            }
        )
    return config


def sentry_init(verbose: bool = True):
    """Инициализация Sentry"""
    config = sentry_get_config()
    if not config.get("dsn"):
        if verbose:
            print("\n[E] Sentry DSN not found, monitoring disabled")
        return False

    if not config.get("debug", False):
        logging.getLogger("sentry_sdk").setLevel(logging.WARNING)

    try:
        sentry_sdk.init(**config)
        if verbose:
            print(f"\n[+] Sentry initialized:")
            print(f"    Environment: {config['environment']}")
            print(f"    Release: {config['release']}")
            print(f"    Traces: {config.get('traces_sample_rate', 'N/A')}")
        return True
    except Exception as e:
        if verbose:
            print(f"\n[E] Sentry initialization error: {e}")
        return False


def sentry_capture_message(
    message: str, level: str = "info", filename: Optional[str] = None, tags: Optional[Dict] = None, extras: Optional[Dict] = None
):
    """Отправка сообщения в Sentry"""
    with sentry_sdk.configure_scope() as scope:
        # Всегда добавляем source file как тег
        if filename:
            scope.set_tag("source_file", filename)
        
        if tags:
            for key, value in tags.items():
                scope.set_tag(key, value)
        if extras:
            for key, value in extras.items():
                scope.set_extra(key, value)
        
        sentry_sdk.capture_message(message, level)


class Logger:
    """Синглтон-логгер с очередью повторных попыток"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.logging_type = os.getenv("LOGGING_TYPE", "local").strip().lower()
        self.log_file = os.getenv("LOG_FILE", "logs/app.log")
        self._local_lock = threading.Lock()
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

        self.sentry_available = False
        self.fallback_reason = None

        # Настройки повторных попыток
        self.retry_enabled = os.getenv("SENTRY_RETRY_ENABLED", "true").lower() == "true"
        self.retry_interval = int(os.getenv("SENTRY_RETRY_INTERVAL", "5"))
        self.max_retries = int(os.getenv("SENTRY_MAX_RETRIES", "3"))

        # Настройки буферизации в файл
        self.buffer_enabled = os.getenv("LOG_BUFFER_ENABLED", "true").lower() == "true"
        self.buffer_file = os.getenv("LOG_BUFFER_FILE", str(Path(self.log_file).parent / "buffer.jsonl"))
        
        self._retry_queue = queue.Queue()
        self._retry_thread = None
        self._stop_retry = threading.Event()

        # Инициализация Sentry если выбран этот тип логирования
        if self.logging_type == "sentry":
            self.sentry_available = sentry_init(verbose=False)
            if not self.sentry_available:
                self.fallback_reason = (
                    "Sentry initialization failed (invalid DSN or connection error)"
                )
                self._write_fallback_info()

        # Запускаем фоновый поток для повторных попыток
        if self.retry_enabled:
            self._retry_thread = threading.Thread(
                target=self._retry_worker, daemon=True
            )
            self._retry_thread.start()
            
            # Восстанавливаем сообщения из буфера при старте
            if self.buffer_enabled:
                self._restore_buffered_messages()

        # Регистрируем обработчики сигналов для graceful shutdown
        self._register_signal_handlers()

    def _register_signal_handlers(self):
        """Регистрация обработчиков сигналов для корректного завершения"""
        def signal_handler(signum, frame):
            self._save_buffered_messages()
            self.close()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(self._save_buffered_messages)

    def _get_buffer_file_path(self) -> Path:
        """Получение пути к файлу буфера"""
        return Path(self.buffer_file)

    def _save_buffered_messages(self):
        """Сохранение сообщений из очереди в файл при завершении"""
        if not self.buffer_enabled:
            return
            
        messages_to_save = []
        while not self._retry_queue.empty():
            try:
                msg = self._retry_queue.get_nowait()
                messages_to_save.append(msg)
            except queue.Empty:
                break
        
        if not messages_to_save:
            return
            
        buffer_path = self._get_buffer_file_path()
        buffer_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with self._local_lock:
                with open(buffer_path, "a", encoding="utf-8") as f:
                    for msg in messages_to_save:
                        f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        except Exception as e:
            sys.stderr.write(f"Failed to save buffered messages: {e}\n")

    def _restore_buffered_messages(self):
        """Восстановление сообщений из файла буфера при старте"""
        buffer_path = self._get_buffer_file_path()
        if not buffer_path.exists():
            return
            
        restored_count = 0
        try:
            with self._local_lock:
                with open(buffer_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            msg = json.loads(line.strip())
                            self._retry_queue.put(msg)
                            restored_count += 1
                        except json.JSONDecodeError:
                            continue
                
                # Очищаем файл буфера после восстановления
                buffer_path.unlink()
                
            if restored_count > 0:
                self._write_local(
                    "INFO",
                    f"Restored {restored_count} buffered messages from previous session",
                    "LOGLIB",
                    is_fallback=False,
                )
        except Exception as e:
            sys.stderr.write(f"Failed to restore buffered messages: {e}\n")

    def _write_fallback_info(self):
        """Запись информации о переходе в fallback режим"""
        if self.fallback_reason:
            timestamp = datetime.now().isoformat()
            line = f"{timestamp} - LOGLIB - WARNING - [FALLBACK ACTIVATED] {self.fallback_reason}\n"
            try:
                with self._local_lock:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(line)
            except Exception as e:
                sys.stderr.write(f"Failed to write fallback info: {e}\n")

    def _write_local(
        self,
        level: str,
        message: str,
        filename: str,
        is_fallback: bool = False,
        fallback_reason: Optional[str] = None,
        tags: Optional[Dict] = None,
        extras: Optional[Dict] = None,
    ):
        """Потокобезопасная запись в локальный файл"""
        timestamp = datetime.now().isoformat()

        if is_fallback:
            reason = fallback_reason or "Sentry unavailable"

            extra_parts = []
            if tags:
                extra_parts.append(f"tags={tags}")
            if extras:
                extra_parts.append(f"extras={extras}")

            extra_str = " ".join(extra_parts)
            if extra_str:
                line = f"{timestamp} - {filename} - {level.upper()} - [FALLBACK: {reason}] ({extra_str}) {message}\n"
            else:
                line = f"{timestamp} - {filename} - {level.upper()} - [FALLBACK: {reason}] {message}\n"
        else:
            line = f"{timestamp} - {filename} - {level.upper()} - {message}\n"

        try:
            with self._local_lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            sys.stderr.write(f"Failed to write log to {self.log_file}: {e}\n")
            sys.stderr.write(line)

    def _retry_worker(self):
        """Фоновый поток для повторных попыток отправки"""
        while not self._stop_retry.is_set():
            try:
                # Проверяем доступность Sentry
                if not self.sentry_available:
                    self.sentry_available = sentry_init(verbose=False)
                    if self.sentry_available:
                        self._write_local(
                            "WARNING",
                            "Sentry connection restored, processing queued logs",
                            "LOGLIB",
                            is_fallback=False,
                        )
                        self.fallback_reason = None

                # Если Sentry доступен, отправляем накопленные сообщения
                if self.sentry_available:
                    sent_count = 0
                    while not self._retry_queue.empty():
                        msg = None
                        try:
                            msg = self._retry_queue.get_nowait()
                            sentry_capture_message(
                                msg["message"],
                                level=msg["level"],
                                tags=msg.get("tags"),
                                extras=msg.get("extras"),
                            )
                            sent_count += 1
                        except queue.Empty:
                            break
                        except Exception as e:
                            # Если ошибка - увеличиваем счетчик попыток
                            if msg is not None:
                                retry_count = msg.get("retry_count", 0) + 1

                                if retry_count < self.max_retries:
                                    msg["retry_count"] = retry_count
                                    self._retry_queue.put(msg)
                                    self._write_local(
                                        "DEBUG",
                                        f"Retry {retry_count}/{self.max_retries} for: {msg['message'][:50]}",
                                        "LOGLIB",
                                        is_fallback=False,
                                    )
                                else:
                                    self._write_local(
                                        "ERROR",
                                        f"Max retries exceeded for: {msg['message'][:50]}",
                                        "LOGLIB",
                                        is_fallback=True,
                                        fallback_reason=f"Max retries ({self.max_retries}) exceeded",
                                    )
                            break

                    if sent_count > 0:
                        self._write_local(
                            "INFO",
                            f"Sent {sent_count} queued messages to Sentry",
                            "LOGLIB",
                            is_fallback=False,
                        )

                time.sleep(self.retry_interval)

            except Exception as e:
                self._write_local(
                    "ERROR", f"Retry worker error: {e}", "LOGLIB", is_fallback=False
                )
                time.sleep(self.retry_interval)

    def log(self, level: str, message: str, filename: str, **kwargs):
        """Основной метод логирования с повторными попытками"""
        tags = kwargs.get("tags")
        extras = kwargs.get("extras")
        force_sentry = kwargs.get("force_sentry", False)

        # ВСЕГДА пишем в локальный файл (полная запись)
        self._write_local(
            level, message, filename, is_fallback=False, tags=tags, extras=extras
        )

        # Если LOCAL режим - только файл, выходим
        if self.logging_type == "local":
            return

        # SENTRY режим: только ошибки, критические и принудительные (lifecycle)
        should_send_sentry = level in ("error", "critical") or force_sentry
        if should_send_sentry:
            if self.sentry_available:
                try:
                    sentry_capture_message(message, level=level, filename=filename, tags=tags, extras=extras)
                except Exception as e:
                    # Ошибка отправки - уже записали в файл
                    pass
            else:
                # Sentry недоступен - добавляем в очередь для повторной отправки
                if self.retry_enabled:
                    self._retry_queue.put(
                        {
                            "level": level,
                            "message": message,
                            "filename": filename,
                            "tags": tags,
                            "extras": extras,
                            "retry_count": 0,
                        }
                    )

    def close(self):
        """Остановка фонового потока"""
        if self._retry_thread and self._retry_thread.is_alive():
            self._stop_retry.set()
            self._retry_thread.join(timeout=5)


_logger = Logger()


def capture_message(level: str, message: str, **kwargs):
    """Основная функция логирования"""
    frame = inspect.currentframe()
    caller_frame = frame.f_back if frame else None
    filename = caller_frame.f_code.co_filename if caller_frame and caller_frame.f_code else "<unknown>"
    filename = os.path.basename(filename)

    _logger.log(level, message, filename, **kwargs)


def shutdown():
    """Явное завершение работы библиотеки"""
    _logger.close()


__version__ = "1.0.0"
