
from __future__ import annotations
import os, json, time, pandas as pd
from typing import Dict, Any
from tqdm import tqdm

from openai import OpenAI
from dataloader import readTourismData

client = OpenAI()                       # relies on OPENAI_API_KEY env. var
SYSTEM_MSG: Dict[str, str] = {
    "role": "system",
    "content": "You are a strict JSON generator. Output only valid JSON."
}
PROMPT_TEMPLATE = """
You are a sustainability analyst.  For the Point-of-Interest below,
return ONLY valid JSON with keys z1–z6 (floats 0–1).

Definitions (copy exactly):
• z1 = estimated CO2-kg per individual visit (lower = greener).
• z2 = current_visitors / carrying_capacity (lower = less crowded).
• z3 = entropy-based seasonality balance (higher = steadier flow).
• z4 = proportion of revenue retained locally (higher = better).
• z5 = crowd-adjusted heritage fragility (lower = safer for culture).
• z6 = overall physical & sensory accessibility (higher = inclusive).

POI:
Name: {name}
Category: {category}
Lat,Lon: {lat}, {lon}
Known sustainability proxy (0-1): {sustainability}
Popularity score (0-1): {popularity}
""".strip()


def _gpt_enrich_row(row: pd.Series) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(**row)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[SYSTEM_MSG, {"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
        timeout=30,
    )
    return json.loads(resp.choices[0].message.content)


# ── public helper – call from Streamlit ───────────────────────────────
def enrich_dataset(data_dir: str = "./data",
                   spinner_callback=None) -> pd.DataFrame:
    """
    • Reads **all** CSVs in `data_dir`
    • Calls GPT for each row that does **not** already have z1–z6
    • Writes:
        – `poi_all_enriched.csv`
        – `poi_<muni>_…_enriched.csv` (one per municipality)
    • Returns the enriched DataFrame
    """
    df = readTourismData(data_dir)
    need = df[[f"z{i}" for i in range(1, 7)]].isna().any(axis=1)

    total = need.sum()
    if total == 0:
        return df      # nothing to do

    # Show progress if we’re inside Streamlit
    pbar = tqdm(total=total, disable=spinner_callback is None)
    if spinner_callback:
        spinner = spinner_callback("Calling GPT…")

    try:
        for idx in df.index[need]:
            try:
                z_vals = _gpt_enrich_row(df.loc[idx])
                for k, v in z_vals.items():
                    df.at[idx, k] = v
            except Exception as e:
                print(f"⚠️ {df.at[idx, 'name']}: {e}")
            finally:
                pbar.update(1)
    finally:
        pbar.close()
        if spinner_callback:
            spinner.empty()       # remove spinner

    # ── save files ────────────────────────────────────────────────────
    master = os.path.join(data_dir, "poi_all_enriched.csv")
    df.to_csv(master, index=False)

    for muni, sub in df.groupby("municipality"):
        fname = os.path.join(data_dir, f"poi_{muni}_enriched.csv")
        sub.to_csv(fname, index=False)

    return df
