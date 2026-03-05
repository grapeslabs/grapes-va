import os
import psycopg2

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DATABASE = os.getenv("PG_DATABASE", "postgres")  # или FR_DB?

# Читаем SQL файл
with open('init_db.sql', 'r', encoding='utf-8') as f:
    sql_script = f.read()

# Подключаемся к PostgreSQL
conn = psycopg2.connect(
    host=PG_HOST,
    port=PG_PORT,
    user=PG_USER,
    password=PG_PASSWORD,
    database=PG_DATABASE
)
conn.autocommit = True
cur = conn.cursor()

# Выполняем скрипт
cur.execute(sql_script)

cur.close()
conn.close()
print("База данных успешно создана!")