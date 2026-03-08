"""
AI Detective — Streamlit App
============================
Task  : Show a researcher two abstracts side-by-side — one written by a human
        (read from a local CSV) and one rewritten by an LLM — and ask them
        to pick the AI-generated one.

Design decisions
-----------------
* Everything lives in st.session_state so the app is completely stateless from
  the server's perspective.  Refreshing the page wipes all history → anonymous.
* No database, no logging, no cookies.
* Score is expressed as a running percentage (correct / total × 100).
* The display order of the two abstracts is randomised each round so the AI
  abstract is not always on the same side.
* The "Show Truth" action is a DOI link that opens in a new browser tab.
* The next round is prefetched in a background thread while the user reads the
  current pair, so "Next" is near-instant.
"""

import os
import random
import threading

import openai
import pandas as pd
import streamlit as st

# ── 1. Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Detective",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 2. Load the local CSV once (cached so it's only read from disk once) ─────
@st.cache_data
def load_papers() -> pd.DataFrame:
    return pd.read_csv("cs_papers.csv")

PAPERS = load_papers()

# ── 3. Session-state initialisation ───────────────────────────────────────────
#    We set every key exactly once so the rest of the code can safely read them.
#    All values are reset to these defaults on a full page refresh.
_DEFAULTS: dict = {
    "score":           0,      # number of rounds the researcher got right
    "total":           0,      # number of rounds attempted so far
    "paper":           None,   # current paper dict
    "human_abstract":  "",     # original abstract text (what the authors wrote)
    "ai_abstract":     "",     # LLM-rewritten abstract text
    "order":           [],     # list like ["human","ai"] or ["ai","human"]
                               #   — determines which slot (A/B) shows which text
    "answered":        False,  # True once the researcher has clicked a guess
    "correct":         None,   # True / False result of the current guess
    "prefetch":        None,   # background-thread slot for the next round
}

for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ── 4. Helper: pick one random paper row from the CSV ────────────────────────
def fetch_random_paper() -> dict | None:
    """Returns a random row from the CSV as a plain dict, or None if empty."""
    if PAPERS.empty:
        return None
    return PAPERS.sample(1).iloc[0].to_dict()


# ── 5. Helper: call LiteLLM proxy to generate an AI abstract ─────────────────
_MODEL = "gpt-oss-120b"
_OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not _OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it to .streamlit/secrets.toml or as an environment variable.")
    st.stop()
_client = openai.OpenAI(
    api_key=_OPENAI_API_KEY,
    base_url="https://ai-research-proxy.azurewebsites.net",
)

def generate_ai_abstract(original: str, title: str) -> str:
    """
    Calls the LiteLLM proxy (OpenAI-compatible) and returns the
    LLM-rewritten abstract.
    """
    system_msg = (
        "You are an expert academic author who writes exactly like a human researcher. "
        "Rewrite the abstract below so it covers the SAME topic, methods, and key findings. "
        "Your rewrite must be virtually indistinguishable from a real human-written abstract. "
        "Follow these rules strictly:\n"
        "- Vary sentence length naturally: mix short punchy sentences with longer complex ones.\n"
        "- Use specific, concrete language — avoid vague hedging words like 'comprehensive', "
        "'significant', 'novel', 'crucial', 'robust', 'noteworthy', 'it is worth noting'.\n"
        "- Do NOT use formulaic transition phrases like 'Furthermore', 'Moreover', 'Additionally', "
        "'In conclusion', 'Notably'.\n"
        "- Include minor stylistic imperfections a real author might have: an occasional "
        "passive-voice slip, a slightly informal word choice, or a parenthetical aside.\n"
        "- Do NOT use em dashes (\u2014) or en dashes (\u2013) anywhere in the text; "
        "use a comma, semicolon, or reword the sentence instead.\n"
        "- Do NOT over-polish. Real abstracts sometimes have awkward phrasing; that's fine.\n"
        "- Mirror the tone and register of the original (e.g. if it's conversational, stay "
        "conversational; if technical, stay technical).\n"
        "- Keep approximately the same length.\n"
        "- Output ONLY the rewritten abstract — no preamble, no commentary, no quotation marks."
    )
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": (
                f"Paper title: {title}\n\n"
                f"Original abstract:\n{original}\n\n"
                "Rewritten abstract:"
            )},
        ],
        temperature=0.7,
        max_tokens=600,
    )
    text = response.choices[0].message.content.strip()
    # Belt-and-suspenders: strip any em/en dashes that slipped past the prompt
    text = text.replace("—", ",").replace("–", "-")
    return text


# ── 6. Prefetch: build the next round in a background thread ──────────────────
def _prefetch_worker(slot: dict) -> None:
    """
    Runs in a daemon thread. Fills *slot* with the next round's data so the
    user experiences near-zero wait when clicking "Next abstract pair".
    """
    try:
        paper = fetch_random_paper()
        if paper is None:
            slot["done"] = True
            return
        human_abstract = str(paper.get("abstract") or "").strip()
        if not human_abstract:
            slot["done"] = True
            return
        title = paper.get("title") or "Untitled"
        ai_abstract = generate_ai_abstract(human_abstract, title)
        slot.update({
            "paper":          paper,
            "human_abstract": human_abstract,
            "ai_abstract":    ai_abstract,
        })
    except Exception as e:
        slot["error"] = str(e)
    finally:
        slot["done"] = True


def kick_off_prefetch() -> None:
    """Spawns a background thread to prepare the next round."""
    slot: dict = {"done": False, "paper": None, "human_abstract": None,
                  "ai_abstract": None, "error": None}
    st.session_state.prefetch = slot
    threading.Thread(target=_prefetch_worker, args=(slot,), daemon=True).start()


# ── 7. Orchestrator: assemble a complete new round ────────────────────────────
def load_new_round() -> None:
    """
    Promotes the prefetched round if it's ready; otherwise fetches synchronously.
    Always kicks off a new background prefetch for the round after this one.
    """
    prefetch = st.session_state.prefetch

    if prefetch and prefetch.get("done") and prefetch.get("paper") and not prefetch.get("error"):
        # Fast path — use the data the background thread already prepared
        paper          = prefetch["paper"]
        human_abstract = prefetch["human_abstract"]
        ai_abstract    = prefetch["ai_abstract"]
    else:
        # Slow path — fetch and generate synchronously
        paper = fetch_random_paper()
        if paper is None:
            st.error("No papers found in cs_papers.csv.")
            return
        human_abstract = str(paper.get("abstract") or "").strip()
        if not human_abstract:
            st.warning("Fetched paper had no readable abstract — retrying…")
            st.rerun()
            return
        title = paper.get("title") or "Untitled"
        ai_abstract = generate_ai_abstract(human_abstract, title)

    # Randomise which slot (A or B) shows the human vs AI abstract.
    order = ["human", "ai"]
    random.shuffle(order)

    # Persist everything for rendering and answer-checking.
    st.session_state.paper          = paper
    st.session_state.human_abstract = human_abstract
    st.session_state.ai_abstract    = ai_abstract
    st.session_state.order          = order
    st.session_state.answered       = False
    st.session_state.correct        = None

    # Immediately start preparing the round after this one in the background.
    kick_off_prefetch()


# ── 8. UI: global styles ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1100px; }

/* ── Header ── */
.lab-header {
    border-bottom: 2px solid #1a1a2e;
    padding-bottom: 0.75rem;
    margin-bottom: 1.5rem;
}
.lab-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #1a1a2e;
    letter-spacing: -0.5px;
    margin: 0;
}
.lab-subtitle {
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 0.2rem;
}

/* ── Score bar ── */
.score-bar {
    display: flex;
    align-items: center;
    gap: 2rem;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.75rem 1.25rem;
    margin-bottom: 1.5rem;
}
.score-item { text-align: center; }
.score-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1a1a2e;
    line-height: 1.2;
}
.score-label {
    font-size: 0.72rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.score-divider {
    width: 1px;
    height: 36px;
    background: #e5e7eb;
}

/* ── Paper title ── */
.paper-meta {
    margin-bottom: 1.25rem;
}
.paper-title {
    font-family: 'Lora', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #111827;
    line-height: 1.5;
    margin-bottom: 0.3rem;
}
.paper-instruction {
    font-size: 0.85rem;
    color: #6b7280;
}

/* ── Abstract cards ── */
.abstract-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.25rem 1.4rem;
    font-family: 'Lora', serif;
    font-size: 0.92rem;
    line-height: 1.75;
    color: #1f2937;
    min-height: 220px;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.abstract-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin-bottom: 0.6rem;
}

/* ── Verdict badges (post-answer) ── */
.badge-ai {
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 6px;
    padding: 0.5rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: #92400e;
    display: inline-block;
    margin-top: 0.4rem;
}
.badge-human {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 6px;
    padding: 0.5rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: #166534;
    display: inline-block;
    margin-top: 0.4rem;
}

/* ── Result banner ── */
.result-correct {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 0 6px 6px 0;
    padding: 0.65rem 1rem;
    font-size: 0.9rem;
    font-weight: 500;
    color: #15803d;
    margin: 1rem 0;
}
.result-wrong {
    background: #fef2f2;
    border-left: 4px solid #ef4444;
    border-radius: 0 6px 6px 0;
    padding: 0.65rem 1rem;
    font-size: 0.9rem;
    font-weight: 500;
    color: #b91c1c;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── 9. Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lab-header">
    <div class="lab-title">AI Detective</div>
    <div class="lab-subtitle">
        Can you tell which abstract was written by AI?
        &nbsp;·&nbsp; Score resets on refresh &nbsp;·&nbsp; Nothing is saved.
    </div>
</div>
""", unsafe_allow_html=True)

# ── 10. Score bar ─────────────────────────────────────────────────────────────
if st.session_state.total > 0:
    pct     = round(100 * st.session_state.score / st.session_state.total)
    correct = st.session_state.score
    total   = st.session_state.total
    wrong   = total - correct
    st.markdown(f"""
    <div class="score-bar">
        <div class="score-item">
            <div class="score-value">{pct}%</div>
            <div class="score-label">Accuracy</div>
        </div>
        <div class="score-divider"></div>
        <div class="score-item">
            <div class="score-value">{correct}</div>
            <div class="score-label">Correct</div>
        </div>
        <div class="score-divider"></div>
        <div class="score-item">
            <div class="score-value">{wrong}</div>
            <div class="score-label">Incorrect</div>
        </div>
        <div class="score-divider"></div>
        <div class="score-item">
            <div class="score-value">{total}</div>
            <div class="score-label">Rounds</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(
        "<p style='color:#9ca3af; font-size:0.85rem; margin-bottom:1.25rem;'>"
        "Make your first pick below to start your score.</p>",
        unsafe_allow_html=True,
    )

# ── 11. Bootstrap: load the very first pair on app start ──────────────────────
if st.session_state.paper is None:
    with st.spinner("Loading first pair…"):
        load_new_round()

# ── 12. Main game area ────────────────────────────────────────────────────────
if st.session_state.paper:
    paper = st.session_state.paper
    title = paper.get("title") or "Untitled"
    raw_doi = paper.get("doi") or ""
    # Normalise DOI to a full URL
    if raw_doi:
        doi_url = raw_doi if raw_doi.startswith("http") else f"https://doi.org/{raw_doi}"
    else:
        doi_url = ""

    order = st.session_state.order
    abstracts = {
        "human": st.session_state.human_abstract,
        "ai":    st.session_state.ai_abstract,
    }

    # Paper title + instruction
    st.markdown(f"""
    <div class="paper-meta">
        <div class="paper-title">{title}</div>
        <div class="paper-instruction">
            Read both abstracts carefully, then select the one you believe was generated by AI.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    for col, label, kind in [
        (col_a, "A", order[0]),
        (col_b, "B", order[1]),
    ]:
        with col:
            st.markdown(
                f"<div class='abstract-label'>Abstract {label}</div>"
                f"<div class='abstract-card'>{abstracts[kind]}</div>",
                unsafe_allow_html=True,
            )

            if not st.session_state.answered:
                if st.button(
                    f"This is the AI abstract",
                    key=f"guess_{label}",
                    use_container_width=True,
                ):
                    correct = (kind == "ai")
                    st.session_state.answered = True
                    st.session_state.correct  = correct
                    st.session_state.total   += 1
                    if correct:
                        st.session_state.score += 1
                    st.rerun()
            else:
                if kind == "ai":
                    st.markdown("<div class='badge-ai'>🤖 AI-generated</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='badge-human'>✍️ Human-written</div>", unsafe_allow_html=True)

    # ── 13. Post-answer feedback + navigation ─────────────────────────────────
    if st.session_state.answered:
        if st.session_state.correct:
            st.markdown("<div class='result-correct'>✓ Correct — you identified the AI-generated abstract.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-wrong'>✗ Incorrect — the AI abstract fooled you this time.</div>", unsafe_allow_html=True)

        st.write("")
        action_col1, action_col2 = st.columns([1, 1])

        with action_col1:
            if doi_url:
                st.link_button(
                    "View Source Paper",
                    url=doi_url,
                    help="Opens the original paper via DOI.",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.markdown("<span style='font-size:0.8rem;color:#9ca3af;'>No DOI available for this paper.</span>", unsafe_allow_html=True)

        with action_col2:
            if st.button("Next Abstract Pair →", use_container_width=True, type="primary"):
                prefetch = st.session_state.prefetch
                if prefetch and prefetch.get("done") and prefetch.get("paper"):
                    load_new_round()
                    st.rerun()
                else:
                    with st.spinner("Generating AI abstract…"):
                        load_new_round()
                    st.rerun()
