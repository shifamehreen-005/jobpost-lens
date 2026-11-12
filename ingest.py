"""Sample job postings, embed them, and store them in Postgres."""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from db import get_connection

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "postings.csv"
SAMPLE_PATH = ROOT / "data" / "sample.csv"
SAMPLE_SIZE = 50
RANDOM_STATE = 42
DESCRIPTION_CHARS = 800
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS postings (
    id bigint PRIMARY KEY,
    title text,
    location text,
    work_type text,
    experience_level text,
    card text,
    company_name text,
    listed_at timestamptz,
    emb_title vector(384),
    emb_full vector(384)
)
"""

INSERT = """
INSERT INTO postings (
    id, title, location, work_type, experience_level,
    card, company_name, listed_at, emb_title, emb_full
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def _text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def load_postings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Move postings.csv into the data folder first."
        )
    df = pd.read_csv(
        path,
        usecols=[
            "job_id",
            "company_name",
            "title",
            "description",
            "location",
            "formatted_work_type",
            "formatted_experience_level",
            "original_listed_time",
        ],
        low_memory=False,
    )
    df["title"] = df["title"].map(_text)
    df["description"] = df["description"].map(_text)
    df = df[df["title"].ne("") & df["description"].ne("")].copy()
    df["listed_at"] = pd.to_datetime(df["original_listed_time"], unit="ms", utc=True)
    df = df.dropna(subset=["job_id", "listed_at"])
    return df


def sample_evenly(df: pd.DataFrame, n: int = SAMPLE_SIZE, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Sort by listed_at and take n rows spaced evenly across that order.

    A seeded shuffle runs first so rows that share a listed_at stay in a
    reproducible order before the even spacing is applied.
    """
    ordered = (
        df.sample(frac=1, random_state=random_state)
        .sort_values("listed_at", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(ordered) <= n:
        return ordered
    positions = np.round(np.linspace(0, len(ordered) - 1, num=n)).astype(int)
    positions = pd.unique(positions)
    return ordered.iloc[positions].reset_index(drop=True)


def build_card(row) -> str:
    description = _text(row["description"])[:DESCRIPTION_CHARS]
    parts = [
        _text(row["title"]),
        _text(row["location"]),
        _text(row["formatted_work_type"]),
        _text(row["formatted_experience_level"]),
        description,
    ]
    return "\n".join(part for part in parts if part)


def embed(sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model = SentenceTransformer(MODEL_NAME)
    emb_title = model.encode(sample["title"].tolist(), convert_to_numpy=True)
    emb_full = model.encode(sample["card"].tolist(), convert_to_numpy=True)
    expected = (len(sample), 384)
    if emb_title.shape != expected or emb_full.shape != expected:
        raise RuntimeError(
            f"Expected embeddings of shape {expected}, got {emb_title.shape} and {emb_full.shape}"
        )
    return emb_title, emb_full


def insert_rows(sample: pd.DataFrame, emb_title: np.ndarray, emb_full: np.ndarray) -> int:
    rows = []
    for position in range(len(sample)):
        row = sample.iloc[position]
        rows.append(
            (
                int(row["job_id"]),
                row["title"],
                _text(row["location"]) or None,
                _text(row["formatted_work_type"]) or None,
                _text(row["formatted_experience_level"]) or None,
                row["card"],
                _text(row["company_name"]) or None,
                row["listed_at"].to_pydatetime(),
                np.asarray(emb_title[position], dtype=np.float32),
                np.asarray(emb_full[position], dtype=np.float32),
            )
        )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE)
            cur.execute("ALTER TABLE postings ADD COLUMN IF NOT EXISTS company_name text")
            cur.executemany(INSERT, rows)
            inserted = len(rows) if cur.rowcount < 0 else cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return inserted


def main() -> None:
    postings = load_postings(CSV_PATH)
    sample = sample_evenly(postings)
    sample["card"] = sample.apply(build_card, axis=1)

    emb_title, emb_full = embed(sample)
    inserted = insert_rows(sample, emb_title, emb_full)

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample[["job_id", "title", "location", "listed_at", "card"]].rename(
        columns={"job_id": "id"}
    ).to_csv(SAMPLE_PATH, index=False)

    listed_min = sample["listed_at"].min().isoformat()
    listed_max = sample["listed_at"].max().isoformat()
    print(f"Inserted {inserted} rows")
    print(f"listed_at min={listed_min} max={listed_max}")


if __name__ == "__main__":
    main()
