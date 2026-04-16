#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль миграций структуры базы данных.

Содержит список миграций и функцию apply_migrations() для их выполнения.
После вызова функции (даже при частичных ошибках) файл переименовывается в .bak,
чтобы при следующем запуске миграции не повторялись.
"""

import os
from typing import List, Tuple, Callable, Optional

# ========== СПИСОК МИГРАЦИЙ ==========
# Формат: (sql, mode, description)
# mode: '' — для всех режимов, 'pacs' — только для pacs, 'pin' — только для pin

migrations: List[Tuple[str, str, str]] = [
    (
        "ALTER TABLE unknown ADD COLUMN IF NOT EXISTS view_percone BOOLEAN DEFAULT true;",
        '',
        "Добавление колонки view_percone в таблицу unknown (неустановленные лица). "
        "При переносе неизвестного лица в известные запись не удаляется, "
        "а получает view_percone = false."
    ),
    (
        "CREATE INDEX IF NOT EXISTS analytics_events_person_photobank_id_idx "
        "ON analytics_events(person_photobank_id);",
        'pacs',
        "Создание индекса по полю person_photobank_id в таблице analytics_events "
        "для ускорения выборки событий по конкретному лицу."
    ),

]


# ========== ЛОГИКА ВЫПОЛНЕНИЯ ==========

def apply_migrations(
    db_connection,                     # экземпляр FRDatabase (должен иметь _get_cursor)
    mode: str,                         # текущий режим ('pacs', 'pin')
    log_func: Optional[Callable[[str, str], None]] = None,
) -> bool:
    """
    Выполняет все подходящие миграции для данного режима.

    Аргументы:
        db_connection — объект FRDatabase (метод _get_cursor обязателен)
        mode          — строка режима (например, 'pacs' или 'pin')
        log_func      — функция логирования с сигнатурой (level, message)
                        Если None, используется print.

    Возвращает:
        True, если файл миграций существовал и был обработан (даже с ошибками),
        False — если файл уже переименован или не найден.
    """
    # Определяем функцию логирования
    if log_func is None:
        def log_func(level, msg):
            print(f"[{level.upper()}] {msg}")

    # Получаем путь к текущему файлу (update_schema.py)
    current_file = os.path.abspath(__file__)

    # Если файл уже переименован (не .py или не существует) — выходим
    if not current_file.endswith('.py') or not os.path.isfile(current_file):
        log_func("debug", f"Файл миграций уже обработан или отсутствует: {current_file}")
        return False

    log_func("info", f"Применение миграций из {os.path.basename(current_file)}...")

    # Фильтруем миграции по режиму
    filtered = []
    for sql, mode_required, desc in migrations:
        if mode_required == '' or mode_required == mode:
            filtered.append((sql, desc))
        else:
            log_func("info", f"Миграция пропущена (режим '{mode_required}' не соответствует '{mode}'): {desc}")

    if not filtered:
        log_func("info", f"Нет миграций для режима '{mode}'")
        # Всё равно переименовываем файл, чтобы при следующем запуске не проверять
        _rename_self(current_file, log_func)
        return True

    log_func("info", f"Найдено {len(filtered)} миграций для режима '{mode}', применяем...")

    success_count = 0
    error_count = 0
    for idx, (sql, desc) in enumerate(filtered, start=1):
        try:
            with db_connection._get_cursor() as cursor:
                cursor.execute(sql)
            log_func("info", f"Миграция {idx}/{len(filtered)}: {desc} — выполнено")
            success_count += 1
        except Exception as e:
            log_func("error", f"Ошибка при выполнении миграции '{desc}': {e}")
            error_count += 1
            # Можно прервать выполнение, раскомментировав raise
            # raise

    if error_count:
        log_func("warning", f"Выполнено {success_count} миграций, ошибок: {error_count}. Файл всё равно будет переименован.")
    else:
        log_func("info", f"Все {success_count} миграций успешно применены.")

    # Переименовываем файл (даже если были ошибки, чтобы не повторять попытки)
    _rename_self(current_file, log_func)
    return True


def _rename_self(file_path: str, log_func: Callable) -> None:
    """Переименовывает текущий файл, добавляя .bak"""
    backup_path = file_path + ".bak"
    try:
        os.rename(file_path, backup_path)
        log_func("info", f"Файл миграций переименован в {os.path.basename(backup_path)}")
    except Exception as e:
        log_func("error", f"Не удалось переименовать файл миграций: {e}")