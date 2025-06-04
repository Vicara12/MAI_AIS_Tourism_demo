from openai import OpenAI

client = OpenAI(api_key=os.getenv("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"))
from tqdm import tqdm    # progress‑bar for long calls


def gpt_enrich_row(row):
    PROMPT_TEMPLATE = """
    You are a sustainability analyst.  For the Point-of-Interest below,
    return ONLY valid JSON with six keys z1–z6 (values 0‒1, floats).

    Definitions (copy exactly):
    • z1 = estimated CO2-kg per individual visit (lower = greener).
    • z2 = current_visitors / carrying_capacity (lower = less crowded).
    • z3 = entropy-based seasonality balance (higher = steadier flow).
    • z4 = proportion of revenue retained in the local economy (higher = better).
    • z5 = crowd-adjusted heritage fragility (lower = safer for culture).
    • z6 = overall physical & sensory accessibility (higher = more inclusive).

    POI:
    Name: {name}
    Category: {category}
    Lat,Lon: {lat}, {lon}
    Known sustainability proxy (0-1): {sustainability}
    Popularity score (0-1): {popularity}

    Return JSON ONLY, nothing else.
    """
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[{"role": "user", "content": PROMPT_TEMPLATE}],
    temperature=0.2)
    data = json.loads(response.choices[0].message.content)
    return data

# --- Batch call (commented out to avoid accidental API usage) ---
indicator_rows = []
for _, r in tqdm(df.iterrows(), total=len(df)):
    indicator_rows.append(gpt_enrich_row(r))
ind = pd.DataFrame(indicator_rows)
df = pd.concat([df.reset_index(drop=True), ind], axis=1)

# --- For demo purposes generate random but reproducible values ---
rng = np.random.default_rng(42)
for z in range(1,7):
    df[f'z{z}'] = rng.random(len(df))
df.head()