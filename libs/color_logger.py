import logging
from logging.handlers import RotatingFileHandler
from colorama import Fore, Back, Style, init
from typing import Dict, Any, Optional
import sys

init(autoreset=True)  # Автоматический сброс стилей после каждого сообщения

    # Цветовые коды для преобразования маркеров
COLOR_MAP = {
        # Цвета текста
        'black': Fore.BLACK,
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE,
        'lightblack': Fore.LIGHTBLACK_EX,
        'lightred': Fore.LIGHTRED_EX,
        'lightgreen': Fore.LIGHTGREEN_EX,
        'lightyellow': Fore.LIGHTYELLOW_EX,
        'lightblue': Fore.LIGHTBLUE_EX,
        'lightmagenta': Fore.LIGHTMAGENTA_EX,
        'lightcyan': Fore.LIGHTCYAN_EX,
        'lightwhite': Fore.LIGHTWHITE_EX,
        'highlight': Fore.LIGHTMAGENTA_EX,  # Специальный маркер

        # Цвета фона
        'bg_black': Back.BLACK,
        'bg_red': Back.RED,
        'bg_green': Back.GREEN,
        'bg_yellow': Back.YELLOW,
        'bg_blue': Back.BLUE,
        'bg_magenta': Back.MAGENTA,
        'bg_cyan': Back.CYAN,
        'bg_white': Back.WHITE,
        'bg_lightblack': Back.LIGHTBLACK_EX,
        'bg_lightred': Back.LIGHTRED_EX,
        'bg_lightgreen': Back.LIGHTGREEN_EX,
        'bg_lightyellow': Back.LIGHTYELLOW_EX,
        'bg_lightblue': Back.LIGHTBLUE_EX,
        'bg_lightmagenta': Back.LIGHTMAGENTA_EX,
        'bg_lightcyan': Back.LIGHTCYAN_EX,
        'bg_lightwhite': Back.LIGHTWHITE_EX,

        # Стили текста
        'bright': Style.BRIGHT,
        'dim': Style.DIM,
        'normal': Style.NORMAL,
        'reset': Style.RESET_ALL,

        # Специальные маркеры
        'end': Style.RESET_ALL,
        '/n' : '\n'
    }


class ColorLogger:
    """
    Класс для цветного логирования в консоль и файл с поддержкой маркеров:
    - {color}текст{end} - цвет текста (например: {red}Ошибка{end})
    - {bg_color}текст{end} - цвет фона (например: {bg_red}Внимание{end})
    - {style}текст{end} - стиль текста (например: {bright}Важно{end})
    """

    def __init__(self, name: str, log_file: str = 'app.log', level: int = logging.INFO):
        """
        Инициализация логгера
        :param name: Имя логгера
        :param log_file: Путь к файлу логов
        :param level: Уровень логирования
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Форматтер для файла (без цветов)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Обработчик для файла с ротацией
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Обработчик для консоли с цветами
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        console_handler.addFilter(self.ColorFilter())
        self.logger.addHandler(console_handler)

    class ColorFilter(logging.Filter):
        """Фильтр для преобразования цветовых маркеров в консольном выводе"""
        def filter(self, record):
            if hasattr(record, 'colored_msg'):
                # Восстанавливаем оригинальное сообщение для файла
                record.msg = record.colored_msg
            return True

    @staticmethod
    def raw2color(msg: str):
        # Заменяем все маркеры цветов
        for marker, color_code in COLOR_MAP.items():
            msg = msg.replace(f"[{marker}]", color_code)
        return msg

    def _process_message(self, msg: str, args: Dict[str, Any]) -> str:
        """
        Обрабатывает сообщение: подставляет переменные и цветовые маркеры
        :param msg: Шаблон сообщения
        :param args: Аргументы для подстановки
        :return: Обработанная строка
        """
        # Подставляем переменные если они есть
        if args:
            try:
                msg = msg % args
            except (KeyError, TypeError):
                pass  # Оставляем сообщение как есть в случае ошибки

        return self.raw2color(msg)

    @staticmethod
    def raw2txt(msg: str):
        # Заменяем все маркеры цветов
        for marker, _ in COLOR_MAP.items():
            msg = msg.replace(f"[{marker}]", "")
        return msg

    def _process_message_txt(self, msg: str, args: Dict[str, Any]) -> str:
        """
        Обрабатывает сообщение: подставляет переменные и цветовые маркеры
        :param msg: Шаблон сообщения
        :param args: Аргументы для подстановки
        :return: Обработанная строка
        """
        # Подставляем переменные если они есть
        if args:
            try:
                msg = msg % args
            except (KeyError, TypeError):
                pass  # Оставляем сообщение как есть в случае ошибки

        return self.raw2txt(msg)



    def _log(self, level: int, msg: str, args: Optional[Dict[str, Any]] = None):
        """
        Основной метод логирования
        :param level: Уровень логирования
        :param msg: Сообщение
        :param args: Аргументы для подстановки
        """
        if args is None:
            args = {}


        # Обрабатываем сообщение для консоли
        colored_msg = self._process_message(msg, args)
        text_msg  = self._process_message_txt(msg, args)

        # Создаем LogRecord с дополнительным полем raw_msg
        record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname=__file__,
            lineno=0,
            msg=text_msg,
            args=(),
            exc_info=None
        )
        setattr(record, 'colored_msg', colored_msg)  # Добавляем оригинальное сообщение
        # Передаем запись в логгер
        self.logger.handle(record)

    # Методы для разных уровней логирования
    def debug(self, msg: str, args: Optional[Dict[str, Any]] = None):
        self._log(logging.DEBUG, msg, args)

    def info(self, msg: str, args: Optional[Dict[str, Any]] = None):
        self._log(logging.INFO, msg, args)

    def warning(self, msg: str, args: Optional[Dict[str, Any]] = None):
        self._log(logging.WARNING, msg, args)

    def error(self, msg: str, args: Optional[Dict[str, Any]] = None):
        self._log(logging.ERROR, msg, args)

    def critical(self, msg: str, args: Optional[Dict[str, Any]] = None):
        self._log(logging.CRITICAL, msg, args)


# Пример использования
if __name__ == "__main__":
    #logger = ColorLogger("my_app")

    logger = ColorLogger("my_app", log_file="app.log", level=logging.DEBUG)

    # Сообщение с выделенными фрагментами
    logger.warning("Внимание! [highlight]Дисковое пространство[end] заканчивается!")

    # Простое сообщение
    logger.info("Стандартное сообщение")

    # Сообщение с динамическими цветами
    error_details = {
        "user": "admin",
        "ip": "192.168.1.1"
    }
    logger.error(
        "Ошибка авторизации для [highlight]%(user)s[end] с IP [highlight]%(ip)s[end]",
        error_details
    )

    # Примеры со всеми цветами
    logger.info("[red]Красный текст[end}")
    logger.info("[green]Зеленый текст[end}")
    logger.info("[blue]Синий текст на [bg_yellow]желтом фоне[end]")
    logger.info("[bright][cyan]Яркий бирюзовый текст[end]")
    logger.info("[lightred]Светло-красный[end} и [bg_lightblue]светло-синий фон[end]")

"""
#Как использовать этот класс:
#Создайте файл color_logger.py с приведенным выше кодом
#Импортируйте класс в вашем проекте:

from color_logger import ColorLogger

# Создаем логгер
logger = ColorLogger("my_app", log_file="app.log", level=logging.DEBUG)

# Примеры использования
logger.info("Информационное сообщение")
logger.warning("Предупреждение: {yellow}Внимание{end}!")
logger.error("Ошибка в модуле {cyan}%(module)s{end}", {"module": "авторизация"})
"""
