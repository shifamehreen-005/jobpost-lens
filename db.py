import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


def get_connection():
    """Load DATABASE_URL from .env and return a psycopg connection with pgvector registered."""
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set. Add it to .env.")
    conn = psycopg.connect(database_url)
    # The vector type has to exist before pgvector can register its adapter.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn
