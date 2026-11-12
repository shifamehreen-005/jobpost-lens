# jobpost-lens

jobpost-lens searches a sample of job postings by meaning. It reads `data/postings.csv`, keeps rows with a title and description, turns `original_listed_time` from Unix milliseconds into `listed_at`, and takes 50 rows spaced evenly across that date range (`random_state=42`). Each row becomes a short card (title, location, work type, experience level, and the description trimmed to 800 characters). `sentence-transformers/all-MiniLM-L6-v2` embeds the title and the card into 384-dimensional vectors, stored in Postgres with pgvector. Search ranks postings by cosine distance, with optional date and work-type filters. Eval compares the title vector and the full-card vector with Hit@1, Hit@5, and MRR.

## How to run

Postgres needs the `vector` extension available. `db.py` creates it if your user is allowed to. Put the connection string in `.env`:

```
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DATABASE
```

Setup, from this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that window, then activate again. In Command Prompt, use `.venv\Scripts\activate.bat` instead.

Ingest. Move the source file first if it is still in the project root:

```powershell
mkdir data
move postings.csv data\postings.csv
python ingest.py
```

Ingest prints how many rows were inserted and the min and max `listed_at`. It also writes `data/sample.csv` with `id`, `title`, `location`, `listed_at`, and `card`.

Search. This prints the top 5 matches on `emb_full` between 2020-01-01 and 2030-01-01:

```powershell
python search.py "remote data analyst SQL"
```

Narrow it with `--start`, `--end`, `--work-type` (for example `Full-time`), `-k`, or `--column emb_title`.

Eval. Both question files need columns `question` and `gold_id`. `gold_id` must be an `id` already stored in `postings`. `data/eval_questions_v1.csv` is the original set (copied wording). `data/eval_questions.csv` is the shorter realistic set.

```powershell
python eval.py
```

Eval scores both files on `emb_title` and `emb_full` over the full `listed_at` range, with no work-type filter and `k=5`. It prints a markdown report and saves the same text to `results.md`.

App. With the virtual environment active:

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints, usually http://localhost:8501. The Search tab lists every posting that matches the sidebar filters, and a query ranks the top results. The Eval results tab shows `results.md`.

![jobpost-lens Search tab](docs/streamlit.png)

## Table schema

```sql
CREATE TABLE postings (
    id bigint PRIMARY KEY,
    title text,
    location text,
    work_type text,
    experience_level text,
    card text,
    listed_at timestamptz,
    emb_title vector(384),
    emb_full vector(384)
);
```

`work_type` and `experience_level` come from `formatted_work_type` and `formatted_experience_level`. Similarity in search results is `1 - (embedding <=> query)`, which is cosine similarity.

## Example query

```powershell
python search.py "remote data analyst SQL"
```

A run prints five rows, similarity rounded to 4 decimals, in this column order:

```
id  title  location  listed_at  similarity
```

`listed_at` is timezone-aware. Higher similarity is a closer match. `python search.py` against an empty `postings` table prints `No matching postings.`

## Results

`python eval.py` writes this table to `results.md`. The date range is the full `listed_at` span in `postings`, with no work-type filter and `k=5`.

| questions | column | Hit@1 | Hit@5 | MRR |
| --- | --- | ---: | ---: | ---: |
| copied wording | emb_title | 0.650 | 0.850 | 0.717 |
| copied wording | emb_full | 1.000 | 1.000 | 1.000 |
| realistic | emb_title | 0.850 | 1.000 | 0.925 |
| realistic | emb_full | 1.000 | 1.000 | 1.000 |

Hit@k is the share of questions whose `gold_id` appears in the top k. MRR is the mean of `1/rank`, or 0 when the gold id is outside the top 5. `results.md` also lists the questions each run missed. Copied wording on `emb_title` missed three gold ids. The other three runs missed none.

## What I learned

- Title-only embeddings (`emb_title`) only see the job title. Full-posting embeddings (`emb_full`) see the card: title, location, work type, experience level, and the description trimmed to 800 characters. On the original questions, which copied phrases from those descriptions, title search scored Hit@1 0.650, Hit@5 0.850, and MRR 0.717, and missed three gold ids. Full-posting search scored 1.000 on every metric, because those phrases were already in the card.
- Rewriting the questions as short job-seeker searches (the kind of role, plus at most a city or a seniority) stopped them from copying posting wording. Title search then improved to Hit@1 0.850, Hit@5 1.000, and MRR 0.925, with every gold id inside the top 5. Full-posting search stayed at 1.000. A query like "night nurse in Gainesville" still matches the full card, because the card already contains the role and the city.
