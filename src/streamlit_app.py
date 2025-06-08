import streamlit as st
from streamlit import session_state as ss
from typing import List
import numpy as np

from dataloader import readTourismData
from userprof import Profile
from introscreen import handleProfiles, renderHeader, renderTabs
from ranking_recommender import (
    compute_ranking, runRecommender, displayResults, W as MCDA_W
)

# ────────────────────────── helpers ──────────────────────────
@st.cache_data(show_spinner=False)
def cached_recomm(df, profiles, w_key):
    """Avoid re-running MCDA unless data / profiles / weights change."""
    return runRecommender(df, profiles)

def set_mcda_weights(p1, p2, p3, p4):
    """Map 4 pillar sliders → 7-dim weight vector used inside ranking."""
    MCDA_W[:] = [
        0.32 * p1, 0.48 * p1, 0.20 * p1,   # P1 → z1,z2,z3
        p2,                                # P2 → z4
        p3,                                # P3 → z5
        0.09 * p4, 0.91 * p4               # P4 → z6,z7
    ]

# ───────────────────────── first run ─────────────────────────
if "page" not in ss:
    ss.data            = readTourismData()
    ss.city_locs       = ss.data.groupby("municipality")[["lat", "lon"]].first().to_dict("index")
    ss.dest_types      = set(ss.data["category"])
    ss.profiles        = {1: Profile()}
    ss.profiles_to_del: List[int] = []
    ss.proc_counter    = 2
    ss.rank_ready      = False
    ss.page            = "input"
    ss.cached_res      = None
    ss.cached_res_key  = None

st.set_page_config(layout="wide", page_title="GreenExplorer")

# ───────────────────────── sidebar ───────────────────────────
with st.sidebar:
    st.header("⚙️  Actions")

    if st.button("🧠  LLM enrich dataset"):
        from enrich import enrich_dataset
        ss.data = enrich_dataset("../data", spinner_callback=st.spinner)
        ss.city_locs = ss.data.groupby("municipality")[["lat", "lon"]].first().to_dict("index")
        st.success("Dataset enriched.")

    with st.expander("⚖️  MCDA pillar weights", expanded=False):
        w_env = st.slider("Environment  (P1)", 0.05, 0.45, 0.25, 0.01)
        w_soc = st.slider("Socio-economic  (P2)", 0.05, 0.25, 0.10, 0.01)
        w_cul = st.slider("Cultural  (P3)", 0.05, 0.25, 0.10, 0.01)
        w_usr = st.slider("User-experience  (P4)", 0.30, 0.70, 0.55, 0.01)
    set_mcda_weights(w_env, w_soc, w_cul, w_usr)

    # Compute ranking
    if st.button("📊  Compute ranking"):
        if "z7" not in ss.data:
            ss.data["z7"] = .5
        ss.data = compute_ranking(ss.data)
        ss.rank_ready   = True
        ss.cached_res   = None          # invalidate cache
        st.success("MCDA ranking ready.")

    nav = st.radio(
        "Navigate",
        ["🧳  Travellers", "🗺  Recommendations"],
        index=0 if ss.page == "input" else 1,
        disabled=not ss.rank_ready,
    )
    ss.page = "input" if nav.startswith("🧳") else "results"

# ───────────────────────── INPUT page ─────────────────────────
if ss.page == "input":
    st.title("Traveller profiles")
    if ss.profiles_to_del:
        handleProfiles()
    renderHeader()
    renderTabs()

# ──────────────────────── RESULTS page ────────────────────────
else:
    if not ss.rank_ready:
        st.warning("Please compute the ranking first.")
        st.stop()

    # cache key = (rounded weight vector, tuple of profile prefs)
    w_key = tuple(MCDA_W.round(4))
    if ss.cached_res is None or ss.cached_res_key != w_key:
        with st.spinner("Running ELECTRE → LSP …"):
            ss.cached_res      = cached_recomm(ss.data, ss.profiles, w_key)["Group"]
            ss.cached_res_key  = w_key

    st.title("Group recommendations")
    displayResults(ss.cached_res)
