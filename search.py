"""Search ingested job postings by meaning, date, and work type."""

import argparse

import numpy as np
import pandas as pd
from psycopg import sql
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

from db import get_connection

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLUMNS = ("emb_full", "emb_title")
_model = None


def _embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def search(query, start_date, end_date, work_type=None, k=5, column="emb_full", conn=None, model=None):
    """Embed query and return the nearest postings inside the date window.

    Rows are ordered by cosine distance on column. similarity is 1 - distance.
    work_type is an exact match against postings.work_type when it is given.
    Pass conn and model to reuse a connection and MiniLM model; otherwise this
    opens its own and closes the connection when finished.
    """
    if column not in COLUMNS:
        raise ValueError(f"column must be one of {COLUMNS}")
    if k < 1:
        raise ValueError("k must be at least 1")

    embedder = _embedder() if model is None else model
    embedding = np.asarray(embedder.encode(query), dtype=np.float32)
    filters = [sql.SQL("listed_at BETWEEN %s AND %s")]
    params = [embedding, start_date, end_date]
    if work_type:
        filters.append(sql.SQL("work_type = %s"))
        params.append(work_type)
    params.extend([embedding, k])

    statement = sql.SQL(
        """
        SELECT id, title, location, listed_at,
               1 - ({column} <=> %s) AS similarity
        FROM postings
        WHERE {filters}
        ORDER BY {column} <=> %s
        LIMIT %s
        """
    ).format(
        column=sql.Identifier(column),
        filters=sql.SQL(" AND ").join(filters),
    )

    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(statement, params)
            rows = cur.fetchall()
        if not own_conn:
            conn.rollback()
    except Exception:
        if not own_conn:
            conn.rollback()
        raise
    finally:
        if own_conn:
            conn.close()

    return pd.DataFrame(rows, columns=["id", "title", "location", "listed_at", "similarity"])


def main():
    parser = argparse.ArgumentParser(description="Search ingested job postings.")
    parser.add_argument("query", help="What the job should be about")
    parser.add_argument("--start", default="2020-01-01", help="Inclusive listed_at start")
    parser.add_argument("--end", default="2030-01-01", help="Inclusive listed_at end")
    parser.add_argument("--work-type", default=None, help="Exact work_type, such as Full-time")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--column", default="emb_full", choices=COLUMNS)
    args = parser.parse_args()

    results = search(
        args.query,
        args.start,
        args.end,
        work_type=args.work_type,
        k=args.k,
        column=args.column,
    )
    if results.empty:
        print("No matching postings.")
        return

    printable = results.copy()
    printable["similarity"] = printable["similarity"].map(lambda value: round(float(value), 4))
    print(printable.to_string(index=False))


if __name__ == "__main__":
    main()
