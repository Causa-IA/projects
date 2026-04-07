import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

def get_db_connection():
    database_url = os.getenv("DATABASE_URL")
    print("Conectando a:", database_url)
    conn = psycopg2.connect(database_url)

    return conn


