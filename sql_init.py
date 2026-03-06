import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DATABASE = os.getenv("PG_DATABASE", "face_recognition_db")

try:
    # Подключаемся к postgres
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        database="postgres",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Создаем базу данных
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{PG_DATABASE}'")
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {PG_DATABASE}")

    cur.close()
    conn.close()

    # Подключаемся к нашей БД
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        database=PG_DATABASE,
    )
    cur = conn.cursor()

    # Читаем SQL, но удаляем строки с ALTER TABLE OWNER
    with open("init_db.sql", "r", encoding="utf-8") as f:
        sql_lines = f.readlines()

    # Фильтруем строки с OWNER
    filtered_sql = []
    for line in sql_lines:
        if "OWNER TO" not in line:
            filtered_sql.append(line)

    sql_script = "".join(filtered_sql)

    # Выполняем
    cur.execute(sql_script)
    conn.commit()

    print("База данных создана!")

except Exception as e:
    print(f"Ошибка: {e}")
