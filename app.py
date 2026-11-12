"""Streamlit UI for job posting search and the saved eval report."""

from datetime import datetime, time, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from db import get_connection
from search import MODEL_NAME, search

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.md"
QUESTIONS_PATH = ROOT / "data" / "eval_questions.csv"
EMBEDDINGS = {"Full posting": "emb_full", "Title only": "emb_title"}
GRID_COLUMNS = 3
EXPERIENCE_LEVELS = {
    "Entry level",
    "Mid-Senior level",
    "Associate",
    "Director",
    "Internship",
    "Executive",
}


@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Connecting to the database...")
def load_connection():
    return get_connection()


def utc_date(value):
    if value.tzinfo is None:
        return value.date()
    return value.astimezone(timezone.utc).date()


def day_bounds(start_day, end_day):
    """Cover each chosen calendar day, including timestamps after midnight."""
    start = datetime.combine(start_day, time.min, tzinfo=timezone.utc)
    end = datetime.combine(end_day, time.max, tzinfo=timezone.utc)
    return start, end


def posting_filters(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT min(listed_at), max(listed_at) FROM postings")
        earliest, latest = cur.fetchone()
        cur.execute(
            """
            SELECT DISTINCT work_type
            FROM postings
            WHERE work_type IS NOT NULL AND btrim(work_type) <> ''
            ORDER BY 1
            """
        )
        work_types = [row[0] for row in cur.fetchall()]
    conn.rollback()
    return earliest, latest, work_types


@st.cache_data(show_spinner="Loading postings...")
def load_filtered_postings(start, end, work_type):
    """Return postings in the date window, newest first.

    work_type is an empty string when every work type should be included.
    """
    conn = load_connection()
    statement = """
        SELECT id, company_name, title, location, work_type, listed_at, card
        FROM postings
        WHERE listed_at BETWEEN %s AND %s
    """
    params = [start, end]
    if work_type:
        statement += " AND work_type = %s"
        params.append(work_type)
    statement += " ORDER BY listed_at DESC"
    with conn.cursor() as cur:
        cur.execute(statement, params)
        rows = cur.fetchall()
    conn.rollback()
    records = []
    for row in rows:
        listed_at = row[5]
        records.append(
            {
                "id": int(row[0]),
                "company": (row[1] or "").strip() or "Company not listed",
                "title": row[2] or "Untitled",
                "location": row[3] or "Location not listed",
                "work_type": row[4] or "Work type not listed",
                "listed": listed_at.strftime("%Y-%m-%d") if listed_at is not None else "Date not listed",
                "card": row[6] or "",
            }
        )
    return records


def clear_search():
    st.session_state.query = ""
    st.session_state.active_query = ""


def apply_colors():
    """Recolor the existing layout. Structure and borders stay as they are."""
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F3EBDD;
            color: #1C2830;
        }
        h1, h2, h3 {
            color: #1E3A34;
        }
        section[data-testid="stSidebar"] {
            background-color: #1E3A34;
        }
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #F6F1E7;
        }
        [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
            background-color: #FFF9F2;
            border-color: #E0D2C0 !important;
        }
        [data-testid="stColumn"] [data-testid="stMarkdownContainer"] strong {
            color: #9A3F24;
        }
        [data-testid="stColumn"] [data-testid="stCaptionContainer"] {
            color: #5C6B73;
        }
        button[data-testid="stBaseButton-primary"] {
            background-color: #C65A2E;
            border-color: #C65A2E;
            color: #FFF8F3;
        }
        button[data-testid="stBaseButton-primary"]:hover {
            background-color: #A84822;
            border-color: #A84822;
            color: #FFF8F3;
        }
        button[data-testid="stBaseButton-secondary"] {
            background-color: #FFF9F2;
            border-color: #1E3A34;
            color: #1E3A34;
        }
        [data-testid="stTextInputRootElement"],
        [data-testid="stTextInputRootElement"] > div,
        [data-baseweb="input"],
        [data-baseweb="select"] > div {
            background-color: #FFF9F2 !important;
            color: #1C2830 !important;
        }
        [data-testid="stTextInput"] input,
        [data-baseweb="input"] input,
        [data-baseweb="select"] input,
        [data-baseweb="select"] span {
            color: #1C2830 !important;
            -webkit-text-fill-color: #1C2830 !important;
            caret-color: #1C2830;
        }
        [data-testid="stTextInput"] input::placeholder {
            color: #7A736A !important;
            -webkit-text-fill-color: #7A736A !important;
        }
        button[data-testid="stTab"],
        button[data-testid="stTab"] p {
            color: #1C2830 !important;
        }
        button[data-testid="stTab"][aria-selected="true"],
        button[data-testid="stTab"][aria-selected="true"] p {
            color: #9A3F24 !important;
        }
        [data-testid="stText"],
        [data-testid="stText"] div,
        [data-testid="stExpanderDetails"],
        [data-testid="stExpanderDetails"] div,
        [data-testid="stExpander"] p {
            color: #1C2830 !important;
        }
        div[data-baseweb="slider"] [role="slider"] {
            background-color: #C65A2E;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def description_text(posting):
    """Drop the title, location, and work type that were prefixed onto the card."""
    skip = {
        posting.get("title") or "",
        posting.get("location") or "",
        posting.get("work_type") or "",
        *EXPERIENCE_LEVELS,
    }
    lines = (posting.get("card") or "").split("\n")
    index = 0
    while index < len(lines) and lines[index].strip() in skip:
        index += 1
    text = "\n".join(lines[index:]).strip()
    return text or "No description stored for this posting."


def draw_card(posting, rank=None, similarity=None):
    """Company on top, role underneath, description behind an expander."""
    role = posting["title"]
    if rank is not None:
        role = f"#{rank} {role}"
    details = [posting["location"], posting["work_type"], posting["listed"]]
    if similarity is not None:
        details.append(f"similarity {similarity:.2f}")
    with st.container(border=True):
        st.markdown(f"**{posting.get('company') or 'Company not listed'}**")
        st.markdown(role)
        st.caption(" · ".join(details))
        with st.expander("Description"):
            st.text(description_text(posting))


def draw_grid(entries):
    """Lay cards out in rows so many postings fit on one screen."""
    for start in range(0, len(entries), GRID_COLUMNS):
        columns = st.columns(GRID_COLUMNS)
        for column, (posting, rank, similarity) in zip(columns, entries[start : start + GRID_COLUMNS]):
            with column:
                draw_card(posting, rank=rank, similarity=similarity)


def render_search(conn, earliest, latest, work_types):
    if earliest is None or latest is None:
        st.info("No postings yet. Run python ingest.py, then reload this page.")
        return

    if "query" not in st.session_state:
        st.session_state.query = ""
    if "active_query" not in st.session_state:
        st.session_state.active_query = ""

    default_start = utc_date(earliest)
    default_end = utc_date(latest)
    picked = st.sidebar.date_input("Listed dates", value=(default_start, default_end))
    if isinstance(picked, tuple) and len(picked) == 2:
        picked = tuple(sorted(picked))
    work_choice = st.sidebar.selectbox("Work type", ["Any", *work_types])
    embedding_choice = st.sidebar.radio("Embedding", list(EMBEDDINGS))
    k = st.sidebar.slider("Results", min_value=1, max_value=10, value=5)

    query_col, search_col, clear_col = st.columns([6, 1, 1], vertical_alignment="bottom")
    with query_col:
        st.text_input("Search", key="query", placeholder="remote data analyst SQL")
    with search_col:
        search_clicked = st.button("Search", type="primary", width="stretch")
    with clear_col:
        st.button("Clear", on_click=clear_search, width="stretch")

    if not st.session_state.query.strip():
        st.session_state.active_query = ""
    elif search_clicked:
        st.session_state.active_query = st.session_state.query.strip()

    if not isinstance(picked, tuple) or len(picked) != 2:
        st.warning("Select a start date and an end date.")
        return

    start, end = day_bounds(picked[0], picked[1])
    work_type = "" if work_choice == "Any" else work_choice
    postings = load_filtered_postings(start, end, work_type)
    by_id = {posting["id"]: posting for posting in postings}
    active_query = st.session_state.active_query

    if not active_query:
        st.markdown(f"Showing {len(postings)} postings")
        if not postings:
            st.info("No postings match these filters.")
            return
        draw_grid([(posting, None, None) for posting in postings])
        return

    try:
        frame = search(
            active_query,
            start,
            end,
            work_type=work_type or None,
            k=k,
            column=EMBEDDINGS[embedding_choice],
            conn=conn,
            model=load_model(),
        )
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return

    st.markdown(f"Top {len(frame)} of {len(postings)} postings for: {active_query}")
    if frame.empty:
        st.info("No matching postings.")
        return
    hits = []
    for rank, row in enumerate(frame.itertuples(index=False), start=1):
        posting = by_id.get(int(row.id))
        if posting is None:
            listed = row.listed_at
            posting = {
                "company": "Company not listed",
                "title": row.title or "Untitled",
                "location": row.location or "Location not listed",
                "work_type": "Work type not listed",
                "listed": listed.strftime("%Y-%m-%d") if listed is not None else "Date not listed",
                "card": "",
            }
        hits.append((posting, rank, float(row.similarity)))
    draw_grid(hits)


def render_eval():
    if RESULTS_PATH.exists():
        st.markdown(RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        st.info("results.md is not available yet. Run python eval.py.")

    st.subheader("Eval questions")
    if not QUESTIONS_PATH.exists():
        st.warning("data/eval_questions.csv was not found.")
        return
    questions = pd.read_csv(QUESTIONS_PATH)
    st.dataframe(questions, width="stretch", hide_index=True)


def main():
    st.set_page_config(page_title="jobpost-lens", layout="wide")
    apply_colors()
    st.title("jobpost-lens")

    conn = None
    earliest = latest = None
    work_types = []
    db_error = None
    try:
        conn = load_connection()
        earliest, latest, work_types = posting_filters(conn)
    except Exception as exc:
        db_error = exc

    search_tab, eval_tab = st.tabs(["Search", "Eval results"])
    with search_tab:
        if db_error is not None:
            st.error(f"Could not read postings: {db_error}")
        else:
            render_search(conn, earliest, latest, work_types)
    with eval_tab:
        render_eval()


if __name__ == "__main__":
    main()
