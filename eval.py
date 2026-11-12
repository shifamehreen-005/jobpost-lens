"""Compare title and full-card embeddings on labeled search questions."""

import sys
from pathlib import Path

import pandas as pd

from db import get_connection
from search import search

ROOT = Path(__file__).resolve().parent
QUESTION_SETS = (
    ("copied wording", ROOT / "data" / "eval_questions_v1.csv"),
    ("realistic", ROOT / "data" / "eval_questions.csv"),
)
RESULTS_PATH = ROOT / "results.md"
COLUMNS = ("emb_title", "emb_full")
K = 5


def load_questions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    questions = pd.read_csv(path)
    required = {"question", "gold_id"}
    missing = required - set(questions.columns)
    if missing:
        raise RuntimeError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    questions = questions.dropna(subset=["question", "gold_id"]).copy()
    questions["question"] = questions["question"].map(lambda value: str(value).strip())
    questions = questions[questions["question"].ne("")]
    questions["gold_id"] = questions["gold_id"].map(lambda value: int(value))
    if questions.empty:
        raise RuntimeError(f"{path.name} has no questions")
    return questions.reset_index(drop=True)


def date_bounds():
    """Return the earliest and latest listed_at so the filter keeps every posting."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT min(listed_at), max(listed_at) FROM postings")
            start, end = cur.fetchone()
    finally:
        conn.close()
    if start is None or end is None:
        raise RuntimeError("postings has no listed_at values. Run ingest.py first.")
    return start, end


def score_question(question: str, gold_id: int, start, end, column: str) -> dict:
    results = search(question, start, end, work_type=None, k=K, column=column)
    found_ids = [int(posting_id) for posting_id in results["id"].tolist()]
    if gold_id in found_ids:
        rank = found_ids.index(gold_id) + 1
        reciprocal = 1.0 / rank
    else:
        rank = None
        reciprocal = 0.0
    return {
        "question": question,
        "gold_id": gold_id,
        "rank": rank,
        "hit1": int(rank == 1),
        "hit5": int(rank is not None and rank <= K),
        "reciprocal": reciprocal,
        "top_ids": found_ids,
    }


def evaluate(questions: pd.DataFrame, start, end, column: str, label: str) -> dict:
    print(f"Evaluating {label} / {column}...", file=sys.stderr)
    rows = [
        score_question(row.question, int(row.gold_id), start, end, column)
        for row in questions.itertuples(index=False)
    ]
    count = len(rows)
    return {
        "label": label,
        "column": column,
        "hit1": sum(row["hit1"] for row in rows) / count,
        "hit5": sum(row["hit5"] for row in rows) / count,
        "mrr": sum(row["reciprocal"] for row in rows) / count,
        "rows": rows,
    }


def missed_lines(rows: list) -> list:
    missed = [row for row in rows if row["rank"] is None]
    if not missed:
        return ["None."]
    lines = []
    for row in missed:
        top_ids = ", ".join(str(posting_id) for posting_id in row["top_ids"]) or "(no results)"
        lines.append(f"- **{row['gold_id']}** — {row['question']}")
        lines.append(f"  - top ids: {top_ids}")
    return lines


def render(start, end, reports: list) -> str:
    lines = [
        "# Retrieval eval",
        "",
        f"Date range {start.isoformat()} to {end.isoformat()}, no work_type filter, k={K}.",
        "Hit@k is the share of questions whose gold_id is in the top k.",
        "MRR is the mean of 1/rank, or 0 when the gold_id is outside the top 5.",
        "",
        "| questions | column | Hit@1 | Hit@5 | MRR |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for report in reports:
        lines.append(
            f"| {report['label']} | {report['column']} | {report['hit1']:.3f} | {report['hit5']:.3f} | {report['mrr']:.3f} |"
        )
    lines.append("")
    for report in reports:
        lines.append(f"## Missed by {report['label']}, {report['column']}")
        lines.append("")
        lines.append("Gold id was not in the top 5.")
        lines.append("")
        lines.extend(missed_lines(report["rows"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    start, end = date_bounds()
    reports = []
    for label, path in QUESTION_SETS:
        questions = load_questions(path)
        for column in COLUMNS:
            reports.append(evaluate(questions, start, end, column, label))
    report = render(start, end, reports)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
