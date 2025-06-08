from __future__ import annotations
import re, numpy as np, pandas as pd
from typing import Dict, List
from geopy.distance import geodesic
import streamlit as st
from pyDecision.algorithm import electre_iii
from userprof import Profile

# ─────────────────────────────────────────────────────────────
# 0 ▸ Regex helpers
# ─────────────────────────────────────────────────────────────
_CAT_RX = {
    "culture":   re.compile(r"(culture|museum|heritage|art|history)",  re.I),
    "nature":    re.compile(r"(nature|park|beach|forest|garden|trail)", re.I),
    "nightlife": re.compile(r"(night|club|bar|pub|music)",             re.I),
}
_DIGITS = re.compile(r"\d+")

# ─────────────────────────────────────────────────────────────
# 1 ▸ z7 – individual & group
# ─────────────────────────────────────────────────────────────
def _pref_score(row, p: Profile) -> float:
    w = dict(culture=p.culture, nature=p.nature, nightlife=p.nlife,
             local=p.local_imp, co2=p.co2)
    denom = sum(w.values()) or 1
    s = {
        "culture":   bool(_CAT_RX["culture"].search(str(row["category"]))),
        "nature":    bool(_CAT_RX["nature"].search(str(row["category"]))),
        "nightlife": bool(_CAT_RX["nightlife"].search(str(row["category"]))),
        "local":     0.5 * (row.get("z4", .5) + (1 - row.get("z5", .5))),
        "co2":       1 - row.get("z1", .5),
    }
    return sum(w[k] * s[k] for k in w) / denom

def z7_individual(df: pd.DataFrame, p: Profile) -> pd.Series:
    return df.apply(lambda r: _pref_score(r, p), axis=1).rename("z7")

def z7_group(indiv: List[pd.Series], eta: float = .3) -> pd.Series:
    mat = np.vstack([s.values for s in indiv])
    return pd.Series(eta * mat.min(0) + (1 - eta) * mat.mean(0),
                     index=indiv[0].index, name="z7")

# ─────────────────────────────────────────────────────────────
# 2 ▸ ELECTRE-III-H + 9-item kernel + LSP
# ─────────────────────────────────────────────────────────────
CRITERIA   = [f"z{i}" for i in range(1, 8)]
CRIT_NAMES = ["CO₂", "Overtourism", "Seasonality",
              "Local-eco", "Cultural fragility",
              "Accessibility", "Pref-fit"]

# default global weight vector – overwritten from main.py via set_mcda_weights
W = np.array([0.08, 0.12, 0.05, 0.10, 0.10, 0.05, 0.50])
Q = np.full(7, .05);  P = np.full(7, .20);  V = np.full(7, .50)
RHO        = 0.5
KERNEL_SZ  = 9

def _electre_rank(df: pd.DataFrame) -> pd.Series:
    M = df[CRITERIA].astype(float).to_numpy()
    M[:, [0, 1, 4]] = 1 - M[:, [0, 1, 4]]          # cost → benefit
    _, _, rank_D, *_ = electre_iii(M, P=P, Q=Q, V=V, W=W, graph=False)

    rmap: Dict[int, int] = {}
    for pos, block in enumerate(rank_D, 1):         # descending
        for tok in block.split(";"):
            if (m := _DIGITS.search(tok)):
                rmap[df.index[int(m.group()) - 1]] = pos
    return pd.Series(rmap, name="electre_rank", dtype=float)

def compute_ranking(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # guarantee full  z1…z7  coverage
    for z in CRITERIA:
        df[z] = df.get(z, .5).fillna(.5)

    df["electre_rank"] = _electre_rank(df)

    # ELECTRE kernel (9 best ranks)
    kernel_idx = (
        df.sort_values("electre_rank")
          .drop_duplicates(subset="name", keep="first")
          .head(KERNEL_SZ)
          .index
    )
    util = ((W / W.sum()) * (df.loc[kernel_idx, CRITERIA] ** RHO)).sum(1) ** (1 / RHO)
    df["U_LSP"] = np.nan
    df.loc[kernel_idx, "U_LSP"] = util

    return df.drop_duplicates("name", keep="first")

# ─────────────────────────────────────────────────────────────
# 3 ▸ Pre-filters
# ─────────────────────────────────────────────────────────────
def _prefilter(df: pd.DataFrame, p: Profile) -> pd.DataFrame:
    sub = df.copy()
    if p.avoid:
        sub = sub[~sub["category"].isin(p.avoid)]
    if p.location and p.max_disp:
        sub["distance_km"] = sub.apply(
            lambda r: geodesic(p.location, (r["lat"], r["lon"])).km, axis=1)
        sub = sub[sub["distance_km"] <= p.max_disp]
    if p.mobility_constr:
        sub = sub[sub["z6"] >= .5]
    return sub

# ─────────────────────────────────────────────────────────────
# 4 ▸ API – group only (spec requirement)
# ─────────────────────────────────────────────────────────────
def runRecommender(df0: pd.DataFrame,
                   profiles: Dict[int, Profile]) -> Dict[str, pd.DataFrame]:
    base = _prefilter(df0, next(iter(profiles.values()))).copy()
    if len(profiles) == 1:
        base["z7"] = z7_individual(base, next(iter(profiles.values())))
    else:
        indiv = [z7_individual(base, p) for p in profiles.values()]
        base["z7"] = z7_group(indiv)
    ranked = compute_ranking(base)
    # keep only the 9-item kernel for display
    kernel = ranked.loc[ranked["U_LSP"].notna()].sort_values(
        ["U_LSP", "electre_rank"], ascending=[False, True])
    return {"Group": kernel}

# ─────────────────────────────────────────────────────────────
# 5 ▸ Streamlit presenter – card grid (no filters here)
# ─────────────────────────────────────────────────────────────

# ─────────  add these imports  ─────────
import os, requests, json, re, numpy as np, pandas as pd   # ← keep existing
from typing import Dict, List

# ─── Google Places v1 constants ─────────────────────────────────────────────
GOOGLE_KEY   = os.getenv("GOOGLE_MAPS_KEY")
TXT_ENDPOINT   = "https://places.googleapis.com/v1/places:searchText"
DETAILS_V1     = "https://places.googleapis.com/v1"
PHOTO_V1       = "https://places.googleapis.com/v1"
_photo_cache:   dict[str | None]        = {}
_place_cache:   dict[str, dict]         = {}      # id  -> minimal details
_text_cache:    dict[str, dict]         = {}      # "name|town" -> textSearch

def _text_search(name: str, town: str) -> dict | None:
    """1-shot text search → returns the *first* place dict (cacheable)."""
    key = f"{name}|{town}"
    if key in _text_cache:
        return _text_cache[key]

    if not GOOGLE_KEY:
        print("NO-KEY : set GOOGLE_MAPS_KEY")
        _text_cache[key] = None
        return None

    body = {"textQuery": f"{name} {town}", "maxResultCount": 1,
            "languageCode": "en"}
    hdr  = {"Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_KEY,
            "X-Goog-FieldMask":
                "places.id,places.displayName,places.formattedAddress,"
                "places.location,places.googleMapsUri,places.rating,"
                "places.userRatingCount,places.photos"}
    r = requests.post(TXT_ENDPOINT, headers=hdr, json=body, timeout=6)
    if r.status_code != 200:
        print("SEARCH ✗", key, r.status_code, r.text[:120])
        _text_cache[key] = None
        return None

    place = r.json().get("places", [{}])[0]
    _text_cache[key] = place or None
    print("SEARCH ✓", key)
    return place or None


def get_place_meta(name: str, town: str) -> dict:
    """
    Return a minimal dict with:
        id, address, rating, reviews, maps_uri, photo_url (≤ 400 px).
    Any missing field is set to None.
    """
    place = _text_search(name, town)
    if not place:                         # graceful fallback
        return dict(address=None, rating=None, reviews=None,
                    maps_uri=None, photo=None)

    pid        = place["id"]
    maps_uri   = place.get("googleMapsUri")
    rating     = place.get("rating")
    reviews    = place.get("userRatingCount")
    address    = place.get("formattedAddress")

    # ---- photo (first one) -------------------------------------------------
    photo_url = None
    if place.get("photos"):
        photo_name = place["photos"][0]["name"]          # places/…/photos/…
        photo_url  = (
            f"{PHOTO_V1}/{photo_name}/media"
            f"?maxHeightPx=400&maxWidthPx=400&key={GOOGLE_KEY}"
        )

    _place_cache[pid] = dict(address=address, rating=rating, reviews=reviews,
                             maps_uri=maps_uri, photo=photo_url)
    return _place_cache[pid]

def get_photo_url(name: str, town: str) -> str | None:
    """
    Return a signed /media photo URL (≤400 px) or None.

    • uses *text search v1*  → places[0].photos[0].name
    • builds the /media URL exactly as the Google “Place Photos (New)” guide
    """
    if not GOOGLE_KEY:
        print("NO-KEY        – set GOOGLE_MAPS_KEY env-var")
        return None

    cache_key = f"{name}|{town}"
    if cache_key in _photo_cache:
        return _photo_cache[cache_key]

    # 1️⃣   v1 text-search  ---------------------------------------------------
    body = {
        "textQuery": f"{name} {town}",
        "maxResultCount": 1,
        "languageCode": "en"
    }
    hdr  = {
        "Content-Type":      "application/json",
        "X-Goog-Api-Key":    GOOGLE_KEY,
        # field mask MUST include 'places.photos' or the array is omitted
        "X-Goog-FieldMask":  "places.id,places.displayName,places.photos"
    }

    try:
        r = requests.post(TXT_ENDPOINT, headers=hdr, json=body, timeout=6)
        if r.status_code != 200:
            print("TEXT HTTP✗  ", cache_key, r.status_code, r.text)
            _photo_cache[cache_key] = None
            return None
        data = r.json()
        photos = data["places"][0]["photos"]
        photo_name = photos[0]["name"]         # places/…/photos/…
    except Exception as e:
        print("TEXT PARSE✗ ", cache_key, repr(e))
        _photo_cache[cache_key] = None
        return None

    # 2️⃣   build /media URL  -------------------------------------------------
    url = (
        f"{PHOTO_V1}/{photo_name}/media"
        f"?maxHeightPx=400&maxWidthPx=400&key={GOOGLE_KEY}"
    )
    print("PHOTO✓       ", cache_key)          # success
    _photo_cache[cache_key] = url
    return url



# ─────────  card renderer  ─────────
def _card(poi: pd.Series):
    """
    Render a single POI card (photo + key meta + direct links)
    ----------------------------------------------------------
      ┌──────────────────────────────┐
      │       photo (140 px)         │
      ├──────────────────────────────┤
      │  POI name                    │
      │  muni — category             │
      │  ★ rating  (#reviews)        │
      │  ELECTRE rank | U_LSP        │
      │  📍 Map • 🚗 Navigate         │
      └──────────────────────────────┘
    """
    meta = get_place_meta(poi["name"], poi["municipality"])

    # ── photo (or grey placeholder) ───────────────────────────
    photo_html = (
        f'<img src="{meta["photo"]}" style="width:100%;height:140px;'
        'object-fit:cover;">'
        if meta["photo"] else
        '<div style="height:140px;display:flex;align-items:center;'
        'justify-content:center;background:#f5f5f5">'
        '<span style="color:#999">No photo</span></div>'
    )

    # ── rating line (optional) ────────────────────────────────
    rating_line = ""
    if meta["rating"]:
        # ★★★★☆ unicode stars scaled to rating (round-half star)
        stars = "★" * int(round(meta["rating"]))
        rating_line = (f'{stars} {meta["rating"]:.1f} '
                       f'({meta["reviews"]} reviews)<br>')

    # ── external links (maps / navigate) ──────────────────────
    links = []
    if meta["maps_uri"]:
        links.append(f'<a href="{meta["maps_uri"]}" target="_blank">📍 Map</a>')
    if "lat" in poi and "lon" in poi:
        nav = (f'https://www.google.com/maps/dir/?api=1&'
               f'destination={poi["lat"]},{poi["lon"]}')
        links.append(f'<a href="{nav}" target="_blank">🚗 Navigate</a>')
    links_html = " • ".join(links)

    # ── assemble full card ────────────────────────────────────
    st.markdown(
        f"""
<div style="border:1px solid #DDD;border-radius:10px;
            overflow:hidden;margin-bottom:12px;width:100%">
  {photo_html}
  <div style="padding:8px 10px;font-size:0.87rem;line-height:1.35">
    <b>{poi['name']}</b><br>
    {poi['municipality']} — {poi['category']}<br>
    {rating_line if rating_line else ""}
    <b>ELECTRE rank:</b> {int(poi['electre_rank'])} |
    <b>U<sub>LSP</sub>:</b> {poi['U_LSP']:.3f}<br>
    {links_html}
  </div>
</div>
""",
        unsafe_allow_html=True
    )






# ─────────────────────────────────────────────────────────────
# 5 ▸ Streamlit presenter – card grid + explanations
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# 5 ▸ Streamlit presenter – card grid + explanations
# ─────────────────────────────────────────────────────────────
def displayResults(group_df: pd.DataFrame, top_n: int = 9) -> None:
    """
    Show the TOP-N *unique* POIs (after ELECTRE → LSP) as:
      • global “why this order” summary  (TOP)
      • a 3-column card grid
      • per-item natural-language explanations (explain_row)
      • interactive pair-wise explainer
    """

    # 1 ▸ keep first occurrence of each name, then take top-N rows
    uniq_df = (
        group_df
        .drop_duplicates(subset="name", keep="first")
        .head(top_n)
    )



    # ── CARD GRID ─────────────────────────────────────────────
    st.subheader(f"Top {len(uniq_df)} recommendations")
    cols = st.columns(3)
    for i, (_, poi) in enumerate(uniq_df.iterrows()):
        with cols[i % 3]:
            _card(poi)

    st.divider()

    # ── PER-ITEM EXPLANATIONS ────────────────────────────────
    st.subheader("ℹ️  Individual utility break-downs")
    for _, r in uniq_df.iterrows():
        st.markdown("• " + explain_row(r))

    # ── PAIR-WISE EXPLAINER ──────────────────────────────────
    with st.expander("🔍  Compare two POIs", expanded=False):
        _interactive_pairwise(uniq_df, "pair_global")





def _interactive_pairwise(df: pd.DataFrame, key_prefix: str):
    cols = st.columns(2)
    with cols[0]:
        left = st.selectbox(
            "POI A", df["name"],
            key=f"{key_prefix}_left")
    with cols[1]:
        right = st.selectbox(
            "POI B", df["name"], index=1,
            key=f"{key_prefix}_right")

    if left != right:
        a = df.loc[df["name"] == left].iloc[0]
        b = df.loc[df["name"] == right].iloc[0]
        st.markdown(pairwise_explain(a, b))


def _lsp_parts(row: pd.Series) -> tuple[float, np.ndarray]:
    vec = row[CRITERIA].to_numpy(float)
    pieces = (W / W.sum()) * (vec ** RHO)
    U = pieces.sum() ** (1 / RHO)
    return U, pieces


def explain_row(row: pd.Series) -> str:
    """One-liner describing main positive & weak drivers of utility U."""
    U, parts = _lsp_parts(row)
    pct = parts / parts.sum() * 100
    best = CRIT_NAMES[int(pct.argmax())]
    worst = CRIT_NAMES[int(pct.argmin())]
    return (f"{row['name']} scores **{U:.3f}**. "
            f"Top driver: *{best}* ({pct.max():.1f} %). "
            f"Least: *{worst}* ({pct.min():.1f} %).")


def pairwise_explain(a: pd.Series, b: pd.Series,
                     eps: float = 0.5, n_terms: int = 3) -> str:
    """Why POI *a* outranks *b* (± gap >= eps pp)."""
    _, pa = _lsp_parts(a);  pct_a = pa / pa.sum() * 100
    _, pb = _lsp_parts(b);  pct_b = pb / pb.sum() * 100
    diff = pct_a - pct_b
    adv = [j for j in diff.argsort()[::-1] if diff[j] >  eps][:n_terms]
    lag = [j for j in diff.argsort()       if diff[j] < -eps][:n_terms]

    lines = [f"#### Why **{a['name']}** outranks **{b['name']}**:"]
    for j in adv:
        lines.append(f"• higher *{CRIT_NAMES[j]}* "
                     f"({pct_a[j]:.1f}% vs {pct_b[j]:.1f}%).")
    for j in lag:
        lines.append(f"• trades off lower *{CRIT_NAMES[j]}* "
                     f"({pct_a[j]:.1f}% vs {pct_b[j]:.1f}%).")
    if len(lines) == 1:
        lines.append(f"• practically tied on every criterion "
                     f"(gaps ≤ {eps} pp).")
    return "\n".join(lines)


def global_summary(df: pd.DataFrame, top_k: int = 10) -> list[str]:
    """One sentence per consecutive pair in the top-k ranking."""
    top = df.nsmallest(top_k, "electre_rank")
    msgs = []
    for i in range(len(top) - 1):
        a, b = top.iloc[i], top.iloc[i + 1]
        _, pac = _lsp_parts(a);  pct_a = pac / pac.sum() * 100
        _, pbc = _lsp_parts(b);  pct_b = pbc / pbc.sum() * 100
        gap = pct_a - pct_b
        j = np.argmax(np.abs(gap))
        direction = "higher" if gap[j] > 0 else "lower"
        msgs.append(f"{i+1}>{i+2}: {a['name']} beats {b['name']} "
                    f"via {direction} *{CRIT_NAMES[j]}* "
                    f"({pct_a[j]:.1f}% vs {pct_b[j]:.1f}%).")
    return msgs


def quick_explain(row: pd.Series) -> str:
    U, part = _lsp_parts(row)
    pct = part / part.sum() * 100
    return (f"**{row['name']}** – U={U:.3f}.  "
            f"↑ *{CRIT_NAMES[pct.argmax()]}* {pct.max():.1f} %, "
            f"↓ *{CRIT_NAMES[pct.argmin()]}* {pct.min():.1f} %.")
