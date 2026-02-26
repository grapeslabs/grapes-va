import psycopg2

# Читаем SQL файл
with open('init_db.sql', 'r',encoding='utf-8') as f:
    sql_script = f.read()

# Подключаемся к PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    user="postgres",
    password="postgres",
    database="postgres"
)
conn.autocommit = True
cur = conn.cursor()

# Выполняем скрипт
cur.execute(sql_script)

cur.close()
conn.close()
print("База данных успешно создана!")