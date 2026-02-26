import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
TARGET_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DATABASE = os.getenv("PG_DATABASE", "face_recognition_db")

try:
    # === 1. Подключаемся как postgres (суперпользователь) ===
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user="postgres",
        password=PG_PASSWORD,
        database="postgres",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # === 2. Проверяем пользователя ===
    cur.execute(f"SELECT 1 FROM pg_roles WHERE rolname = '{TARGET_USER}'")
    if not cur.fetchone():
        print(f"Создание пользователя {TARGET_USER}...")
        cur.execute(f"CREATE USER {TARGET_USER} WITH PASSWORD '{PG_PASSWORD}'")
    else:
        print(f"Пользователь {TARGET_USER} уже существует")

    # === 3. Создаём базу данных ===
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{PG_DATABASE}'")
    if not cur.fetchone():
        print(f"Создание базы данных {PG_DATABASE}...")
        cur.execute(f"CREATE DATABASE {PG_DATABASE} OWNER {TARGET_USER}")
    else:
        print(f"База данных {PG_DATABASE} уже существует")
        cur.execute(f"ALTER DATABASE {PG_DATABASE} OWNER TO {TARGET_USER}")

    # === 4. Подключаемся к нашей БД (всё ещё как postgres) ===
    print(f"Подключение к {PG_DATABASE} как postgres...")
    cur.close()
    conn.close()

    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user="postgres",  # Всё ещё postgres!
        password=PG_PASSWORD,
        database=PG_DATABASE,
    )
    cur = conn.cursor()

    # === 5. Создаём расширение vector (только суперпользователь) ===
    print("Создание расширения vector...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # === 6. Читаем SQL и удаляем строки с OWNER TO ===
    with open("init_db.sql", "r", encoding="utf-8") as f:
        sql_lines = f.readlines()

    filtered_sql = []
    for line in sql_lines:
        if "OWNER TO" not in line:
            filtered_sql.append(line)

    sql_script = "".join(filtered_sql)

    # === 7. Выполняем SQL (всё ещё как postgres) ===
    print("Создание таблиц и функций...")
    cur.execute(sql_script)

    # === 8. Передаём права на таблицы пользователю ===
    print(f"Передача прав пользователю {TARGET_USER}...")
    cur.execute(
        f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {TARGET_USER};"
    )
    cur.execute(
        f"GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {TARGET_USER};"
    )
    cur.execute(
        f"GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO {TARGET_USER};"
    )

    # Делаем пользователя владельцем всех таблиц
    tables = ["percone", "cameras", "photo", "unknown", "analytics_events"]
    for table in tables:
        cur.execute(f"ALTER TABLE public.{table} OWNER TO {TARGET_USER};")

    conn.commit()
    cur.close()
    conn.close()

    print(
        f"✅ База данных успешно создана, расширение установлено, права переданы {TARGET_USER}!"
    )

except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback

    traceback.print_exc()
