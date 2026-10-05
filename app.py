import os
import re

import streamlit as st

# --- Bridge Streamlit Cloud secrets -> env vars (must run BEFORE importing agents/tools) ---
for _key in ("GROQ_API_KEY", "TAVILY_API_KEY"):
    try:
        if _key in st.secrets:
            os.environ.setdefault(_key, st.secrets[_key])
    except Exception:
        pass  # no secrets file locally -> .env is used via load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents import (
    build_reader_agent,
    build_search_agent,
    critic_chain,
    llm,
    writer_chain,
)

st.set_page_config(page_title="Research Agent", page_icon="🔷", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{
  --bg:#04060B; --panel:#0A1120; --panel-2:#0F1830;
  --line:rgba(76,125,255,.24); --blue:#2F6BFF; --blue-2:#6AA3FF; --cyan:#3DD6FF;
  --text:#E8EEFF; --muted:#8D9BC0;
}
html, body, .stApp, [data-testid="stMarkdownContainer"], button, textarea, input{
  font-family:'IBM Plex Sans', system-ui, sans-serif !important;
}
.stApp{ background:var(--bg); color:var(--text); }
.stApp::before{
  content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
  background:
    radial-gradient(40% 35% at 18% 12%, rgba(47,107,255,.30), transparent 70%),
    radial-gradient(34% 30% at 86% 8%, rgba(61,214,255,.14), transparent 70%),
    radial-gradient(46% 40% at 72% 96%, rgba(47,107,255,.20), transparent 70%);
  animation:drift 24s ease-in-out infinite alternate;
}
@keyframes drift{ to{ transform:translate3d(-3%,4%,0) scale(1.08);} }
[data-testid="stAppViewContainer"]{ background:transparent; position:relative; z-index:1; }
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stBottom"] > div{ background:transparent; }
#MainMenu, footer, .stDeployButton{ visibility:hidden; }
.block-container{ max-width:1020px; padding-top:2.4rem; padding-bottom:7rem; }

/* sidebar */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#07112A 0%,#04060B 70%);
  border-right:1px solid var(--line); z-index:2;
}
.brand{ display:flex; align-items:center; gap:10px; font:700 1.25rem 'Sora',sans-serif; color:#fff; letter-spacing:-.01em; }
.brand i{ width:14px; height:14px; border-radius:4px; background:linear-gradient(135deg,var(--cyan),var(--blue));
  box-shadow:0 0 16px rgba(61,214,255,.7); display:inline-block; transform:rotate(45deg); }
.side-note{ color:var(--muted); font-size:.85rem; margin:.4rem 0 1rem; }
.topic-card{ padding:12px 14px; border-radius:12px; border:1px solid var(--line);
  background:rgba(47,107,255,.10); color:var(--text); font-size:.92rem; line-height:1.4; }

/* hero */
.hero{ padding:2.2rem 0 1.4rem; animation:rise .8s cubic-bezier(.2,.7,.2,1) both; }
.hero h1{
  font:700 clamp(2.1rem,4.6vw,3.5rem)/1.08 'Sora',sans-serif; letter-spacing:-.035em; margin:0 0 14px;
  background:linear-gradient(100deg,#fff 20%,var(--blue-2) 70%,var(--cyan));
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hero p{ color:var(--muted); font-size:1.08rem; max-width:620px; line-height:1.6; margin:0; }
@keyframes rise{ from{ opacity:0; transform:translateY(14px);} to{ opacity:1; transform:none;} }

/* pipeline stepper */
.pipeline{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0 8px; }
.step{ display:flex; gap:12px; align-items:center; padding:14px 16px; border-radius:14px;
  border:1px solid var(--line); background:rgba(10,17,32,.72); color:var(--muted);
  transition:border-color .3s, box-shadow .3s, color .3s; }
.step .num{ flex:none; width:32px; height:32px; border-radius:50%; display:grid; place-items:center;
  border:1px solid var(--line); font:600 .85rem 'Sora',sans-serif; }
.step b{ display:block; font:600 .95rem 'Sora',sans-serif; margin-bottom:2px; }
.step small{ font-size:.78rem; opacity:.85; line-height:1.3; display:block; }
.step.active{ color:var(--text); border-color:var(--blue);
  box-shadow:0 0 0 1px var(--blue), 0 0 30px rgba(47,107,255,.5); animation:pulse 1.6s ease-in-out infinite; }
.step.active .num{ background:var(--blue); border-color:var(--blue); color:#fff; }
.step.done{ color:var(--text); border-color:rgba(61,214,255,.45); }
.step.done .num{ background:rgba(61,214,255,.14); border-color:var(--cyan); color:var(--cyan); }
@keyframes pulse{ 50%{ box-shadow:0 0 0 1px var(--blue-2), 0 0 44px rgba(47,107,255,.75);} }
@media (max-width:820px){ .pipeline{ grid-template-columns:1fr 1fr; } }

/* stats strip */
.stats{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:0 0 18px; }
.stat{ padding:14px 18px; border-radius:14px; border:1px solid var(--line);
  background:linear-gradient(160deg,rgba(47,107,255,.18),rgba(10,17,32,.6)); }
.stat .v{ font:700 1.6rem 'Sora',sans-serif; color:#fff; letter-spacing:-.02em; }
.stat .v span{ font-size:.95rem; color:var(--muted); font-weight:500; }
.stat .l{ color:var(--muted); font-size:.82rem; margin-top:2px; }

/* chat */
[data-testid="stChatMessage"]{
  background:rgba(10,17,32,.74); border:1px solid var(--line); border-radius:18px;
  padding:18px 22px; margin-bottom:14px; backdrop-filter:blur(10px);
  animation:rise .45s cubic-bezier(.2,.7,.2,1) both;
}
[data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2, [data-testid="stChatMessage"] h3{
  font-family:'Sora',sans-serif; letter-spacing:-.02em; color:#fff; }
[data-testid="stChatMessage"] a{ color:var(--blue-2); }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li{ line-height:1.7; }

[data-testid="stChatInput"]{
  background:var(--panel); border:1px solid var(--line); border-radius:16px;
  box-shadow:0 10px 40px rgba(0,0,0,.5); transition:border-color .25s, box-shadow .25s;
}
[data-testid="stChatInput"]:focus-within{ border-color:var(--blue);
  box-shadow:0 0 0 3px rgba(47,107,255,.25), 0 0 36px rgba(47,107,255,.35); }
[data-testid="stChatInput"] textarea{ color:var(--text); }

/* expanders */
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:12px; background:rgba(4,6,11,.5); margin-top:8px; }
[data-testid="stExpander"] summary{ color:var(--text); }

/* buttons */
.stButton > button, .stDownloadButton > button{
  border-radius:12px; border:1px solid var(--line); background:var(--panel-2); color:var(--text);
  transition:transform .2s, border-color .2s, box-shadow .2s, background .2s; font-weight:500;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  border-color:var(--blue); transform:translateY(-1px); color:#fff;
  box-shadow:0 6px 24px rgba(47,107,255,.35);
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,#2F6BFF,#1A47D8); border-color:#4D82FF; color:#fff;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover{
  box-shadow:0 8px 30px rgba(47,107,255,.65);
}
.hint{ color:var(--muted); font-size:.88rem; margin:18px 0 8px; }

/* misc */
[data-testid="stSpinner"] i{ border-top-color:var(--blue) !important; }
::-webkit-scrollbar{ width:10px; } ::-webkit-scrollbar-thumb{ background:#16244A; border-radius:8px; }
@media (prefers-reduced-motion:reduce){ *, *::before{ animation:none !important; transition:none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Follow-up chain: answers questions using the report + gathered research
# ----------------------------------------------------------------------------
followup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant continuing a conversation about the topic: {topic}.\n"
            "Answer using the report and research below. If the answer isn't covered, say so "
            "clearly instead of guessing. Be concise, factual, and cite URLs when relevant.\n\n"
            "REPORT:\n{report}\n\nRESEARCH:\n{research}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)
followup_chain = followup_prompt | llm | StrOutputParser()

# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
STEPS = [
    ("Search", "Finds recent sources on the web"),
    ("Read", "Scrapes the most relevant page"),
    ("Write", "Drafts a structured report"),
    ("Critique", "Scores it and flags gaps"),
]
EXAMPLES = [
    "Impact of AI on healthcare in 2026",
    "State of solid-state batteries",
    "India's semiconductor mission",
]


def stepper_html(active: int) -> str:
    """active: index of running step; 4 = all done; -1 = idle."""
    parts = []
    for i, (name, desc) in enumerate(STEPS):
        cls = "done" if i < active else "active" if i == active else ""
        mark = "✓" if i < active else str(i + 1)
        parts.append(
            f'<div class="step {cls}"><div class="num">{mark}</div>'
            f"<div><b>{name}</b><small>{desc}</small></div></div>"
        )
    return '<div class="pipeline">' + "".join(parts) + "</div>"


def stats_html(extras: dict, report: str) -> str:
    m = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", extras.get("feedback", ""))
    score = f'{m.group(1)}<span> / 10</span>' if m else "–"
    sources = len(set(re.findall(r"https?://[^\s)\]]+", extras.get("search_results", ""))))
    words = f"{len(report.split()):,}"
    tiles = [
        (score, "Critic score"),
        (str(sources), "Sources found"),
        (words, "Words in report"),
    ]
    return '<div class="stats">' + "".join(
        f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in tiles
    ) + "</div>"


AVATARS = {"user": ":material/person:", "assistant": ":material/neurology:"}


def render_message(msg):
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg.get("kind") == "error":
            st.error(msg["content"])
            return
        extras = msg.get("extras") or {}
        if msg.get("kind") == "report":
            st.markdown(stats_html(extras, msg["content"]), unsafe_allow_html=True)
        st.markdown(msg["content"])
        if extras.get("feedback"):
            with st.expander("Critic's review"):
                st.markdown(extras["feedback"])
        if extras.get("search_results"):
            with st.expander("Search results"):
                st.text(extras["search_results"])
        if extras.get("scraped_content"):
            with st.expander("Scraped content"):
                st.text(extras["scraped_content"])


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("topic", None)
    st.session_state.setdefault("research", "")
    st.session_state.setdefault("report", "")


def reset_chat():
    for k in ("messages", "topic", "research", "report", "pending"):
        st.session_state.pop(k, None)


def build_history():
    history = []
    for m in st.session_state.messages:
        if m.get("kind") != "followup":
            continue
        cls = HumanMessage if m["role"] == "user" else AIMessage
        history.append(cls(content=m["content"]))
    return history


# ----------------------------------------------------------------------------
# Pipeline (same 4 steps as pipeline.py, with a live stepper)
# ----------------------------------------------------------------------------
def run_pipeline_ui(topic: str) -> dict:
    state = {}
    ph = st.empty()
    ph.markdown(stepper_html(0), unsafe_allow_html=True)

    search_result = build_search_agent().invoke(
        {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]}
    )
    state["search_results"] = search_result["messages"][-1].content

    ph.markdown(stepper_html(1), unsafe_allow_html=True)
    reader_result = build_reader_agent().invoke(
        {
            "messages": [
                (
                    "user",
                    f"Based on the following search results about '{topic}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{state['search_results'][:800]}",
                )
            ]
        }
    )
    state["scraped_content"] = reader_result["messages"][-1].content

    ph.markdown(stepper_html(2), unsafe_allow_html=True)
    state["research"] = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )
    state["report"] = writer_chain.invoke({"topic": topic, "research": state["research"]})

    ph.markdown(stepper_html(3), unsafe_allow_html=True)
    state["feedback"] = critic_chain.invoke({"report": state["report"]})

    ph.markdown(stepper_html(4), unsafe_allow_html=True)
    return state


def answer_followup(question: str, search_again: bool) -> str:
    research = st.session_state.research
    if search_again:
        with st.spinner("Searching the web for fresh sources..."):
            extra = build_search_agent().invoke(
                {
                    "messages": [
                        (
                            "user",
                            f"Find detailed, reliable information to answer: {question} "
                            f"(context topic: {st.session_state.topic})",
                        )
                    ]
                }
            )["messages"][-1].content
        research = f"{research}\n\nADDITIONAL RESEARCH FOR FOLLOW-UP:\n{extra}"
        st.session_state.research = research

    return followup_chain.invoke(
        {
            "topic": st.session_state.topic,
            "report": st.session_state.report[:6000],
            "research": research[:6000],
            "history": build_history(),
            "question": question,
        }
    )


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
init_state()

with st.sidebar:
    st.markdown('<div class="brand"><i></i>Research Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-note">Search, read, write and critique, then keep asking.</div>',
                unsafe_allow_html=True)

    if st.session_state.topic:
        st.markdown("**Current topic**")
        st.markdown(f'<div class="topic-card">{st.session_state.topic}</div>', unsafe_allow_html=True)
        st.write("")
        search_again = st.toggle(
            "Search the web for follow-ups",
            value=False,
            help="Off: answers from the existing report. On: runs a fresh search for each question.",
        )
        if st.session_state.report:
            st.download_button(
                "Download report (.md)",
                data=st.session_state.report,
                file_name="research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
    else:
        search_again = False

    if st.button("New chat", use_container_width=True, type="primary", icon=":material/add:"):
        reset_chat()
        st.rerun()

placeholder = (
    "Ask a follow-up about this topic..." if st.session_state.topic else "Enter a research topic..."
)
prompt = st.chat_input(placeholder)
if not prompt:
    prompt = st.session_state.pop("pending", None)

if not st.session_state.messages and not prompt:
    st.markdown(
        '<div class="hero"><h1>Research any topic, backed by real sources</h1>'
        "<p>Four agents search the web, read the best page, write a structured report and critique it. "
        "Then keep asking follow-up questions on the same topic.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(stepper_html(-1), unsafe_allow_html=True)
    st.markdown('<div class="hint">Try one of these</div>', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

for m in st.session_state.messages:
    render_message(m)

if prompt:
    is_first = st.session_state.topic is None
    user_msg = {"role": "user", "content": prompt, "kind": "topic" if is_first else "followup"}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    try:
        if is_first:
            st.session_state.topic = prompt
            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                result = run_pipeline_ui(prompt)
            st.session_state.research = result["research"]
            st.session_state.report = result["report"]
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "kind": "report",
                    "content": result["report"],
                    "extras": {
                        "feedback": result["feedback"],
                        "search_results": result["search_results"],
                        "scraped_content": result["scraped_content"],
                    },
                }
            )
        else:
            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                with st.spinner("Thinking..."):
                    answer = answer_followup(prompt, search_again)
            st.session_state.messages.append({"role": "assistant", "kind": "followup", "content": answer})
    except Exception as e:
        if is_first:
            st.session_state.topic = None
        st.session_state.messages.append(
            {"role": "assistant", "kind": "error",
             "content": f"The pipeline failed: {e}. Check your API keys and try again."}
        )

    st.rerun()