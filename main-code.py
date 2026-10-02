import itertools
import json
import os
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests
from dotenv import load_dotenv


############################################
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "articles.json"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "output"

QUERY = '"midterm elections"'
FROM_DATE = "2026-01-01"
TO_DATE = "2026-09-30"

# Tags that mirror the search and connect to everything.
GENERIC_TAGS = {
    "US midterm elections 2026", "US politics",
    "US news", "World news", "UK news", "News",
    "Donald Trump", "Trump administration", "Republicans", 
    "Democrats", "West Coast"
}

MIN_TAG_ARTICLES = 2     # drop tags appearing on fewer articles than this
MIN_EDGE_WEIGHT = 1      # drop co-occurrences weaker than this

for d in (DATA_FILE.parent, FIG_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

#########STEP 1: fetch from the Guardian API 
def fetch_articles():
    load_dotenv()
    key = os.getenv("GUARDIAN_API_KEY")
    if not key:
        raise SystemExit("GUARDIAN_API_KEY not found. Check your .env file.")

    url = "https://content.guardianapis.com/search"
    rows, page = [], 1
    while True:
        resp = requests.get(url, params={
            "api-key": key,
            "q": QUERY,
            "from-date": FROM_DATE,
            "to-date": TO_DATE,
            "show-tags": "keyword",
            "page-size": 50,
            "page": page,
        }, timeout=30)
        resp.raise_for_status()
        body = resp.json()["response"]

        for a in body["results"]:
            rows.append({
                "id": a["id"],
                "title": a["webTitle"],
                "date": a["webPublicationDate"],
                "section": a["sectionName"],
                "tags": [t["webTitle"] for t in a.get("tags", [])],
            })

        print(f"page {page}/{body['pages']}  (total reported: {body['total']})")
        if page >= body["pages"]:
            break
        page += 1
        time.sleep(0.5)

    return rows


def load_or_fetch():
    if DATA_FILE.exists():
        print(f"Loading cached data from {DATA_FILE}")
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    rows = fetch_articles()
    DATA_FILE.write_text(json.dumps(rows), encoding="utf-8")
    print(f"Saved {len(rows)} articles to {DATA_FILE}")
    return rows

########STEP 2: inspect the data (use this to pick GENERIC_TAGS)
def inspect(rows):
    ids = [r["id"] for r in rows]
    print(f"\nArticles: {len(rows)}  |  duplicate IDs: {len(ids) - len(set(ids))}")
    print(f"Articles with no tags: {sum(1 for r in rows if not r['tags'])}")
    counts = Counter(t for r in rows for t in set(r["tags"]))
    print(f"Unique tags: {len(counts)}")
    print("\nTop 25 tags (look for generic ones to add to GENERIC_TAGS):")
    for tag, c in counts.most_common(25):
        print(f"  {c:5d}  {tag}")
    print("\nArticles by section:")
    print(pd.Series([r["section"] for r in rows]).value_counts().head(8))

#########STEP 3: build the co-occurrence graph
# ----------------------------------------------------------------------
def build_graph(rows, min_tag=MIN_TAG_ARTICLES, min_edge=MIN_EDGE_WEIGHT):
    tag_counts = Counter(t for r in rows for t in set(r["tags"]) - GENERIC_TAGS)
    keep = {t for t, c in tag_counts.items() if c >= min_tag}
    pairs = Counter()
    for r in rows:
        tags = sorted((set(r["tags"]) - GENERIC_TAGS) & keep)
        pairs.update(itertools.combinations(tags, 2))
    G = nx.Graph()
    for (a, b), w in pairs.items():
        if w >= min_edge:
            G.add_edge(a, b, weight=w, distance=1 / w)
    nx.set_node_attributes(G, {t: tag_counts[t] for t in G}, "articles")
    return G, pairs

def sensitivity(rows, k=5):
    results = {}
    for min_tag, min_edge in [(2, 1), (3, 2), (5, 3)]:
        G, _ = build_graph(rows, min_tag, min_edge)
        core = G.subgraph(max(nx.connected_components(G), key=len))
        bt = nx.betweenness_centrality(core, weight="distance")
        st = dict(core.degree(weight="weight"))
        results[(min_tag, min_edge)] = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "top_strength": sorted(st, key=st.get, reverse=True)[:k],
            "top_betweenness": sorted(bt, key=bt.get, reverse=True)[:k],
        }
    out = pd.DataFrame(results).T
    out.to_csv(OUT_DIR / "sensitivity.csv")
    print(out.to_string())

##################STEP 4: centrality measures
# ----------------------------------------------------------------------
def compute_centrality(G):
    # closeness/betweenness are only meaningful within one connected piece
    core_nodes = max(nx.connected_components(G), key=len)
    core = G.subgraph(core_nodes).copy()
    print(f"Largest component: {core.number_of_nodes()} of "
          f"{G.number_of_nodes()} nodes "
          f"({100 * core.number_of_nodes() / G.number_of_nodes():.0f}%)")

    table = pd.DataFrame({
        "articles": nx.get_node_attributes(core, "articles"),
        "strength": dict(core.degree(weight="weight")),
        "degree_centrality": nx.degree_centrality(core),
        "betweenness": nx.betweenness_centrality(core, weight="distance"),
        "closeness": nx.closeness_centrality(core, distance="distance"),
    })
    table.index.name = "topic"
    table.to_csv(OUT_DIR / "centrality.csv")

    for col in ["strength", "betweenness", "closeness"]:
        print(f"\nTop 10 by {col}")
        print(table.sort_values(col, ascending=False)[[col]].head(10).round(4))
    return core, table


def top_pairs(pairs, n=10):
    df = pd.DataFrame(
        [(a, b, w) for (a, b), w in pairs.most_common(n)],
        columns=["topic_a", "topic_b", "articles_together"],
    )
    df.to_csv(OUT_DIR / "top_pairs.csv", index=False)
    print(f"\nTop {n} topic pairs")
    print(df)
    return df


# STEP 5: figures
# ----------------------------------------------------------------------
def plot_network(core, table, n=40):
    top = table.sort_values("strength", ascending=False).head(n).index
    H = core.subgraph(top)
    pos = nx.spring_layout(H, weight="weight", seed=42)
    plt.figure(figsize=(12, 9))
    nx.draw_networkx(
        H, pos, font_size=7, node_color="#4C78A8",
        node_size=[table.loc[x, "strength"] * 2 for x in H],
        width=[H[u][v]["weight"] * 0.15 for u, v in H.edges()],
        edge_color="lightgray",
    )
    plt.axis("off")
    plt.savefig(FIG_DIR / "network.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_heatmap(pairs, table, n=20):
    top = list(table.sort_values("strength", ascending=False).head(n).index)
    m = pd.DataFrame(0, index=top, columns=top)
    for (a, b), w in pairs.items():
        if a in m.index and b in m.index:
            m.loc[a, b] = m.loc[b, a] = w
    plt.figure(figsize=(10, 8))
    plt.imshow(m.values, cmap="Blues")
    plt.colorbar(label="Articles in common")
    plt.xticks(range(n), top, rotation=90, fontsize=7)
    plt.yticks(range(n), top, fontsize=7)
    plt.savefig(FIG_DIR / "heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_monthly(rows):
    s = pd.to_datetime([r["date"] for r in rows]).tz_localize(None).to_period("M").value_counts().sort_index()
    s.plot(kind="bar", figsize=(8, 4))
    plt.ylabel("Articles")
    plt.title("Articles per month")
    plt.savefig(FIG_DIR / "monthly.png", dpi=200, bbox_inches="tight")
    plt.close()

#merge congress tags into one
CONGRESS_TAGS = {"US Senate", "House of Representatives"}

def merge_congress(rows):
    merged = []
    for r in rows:
        tags = ["US Congress" if t in CONGRESS_TAGS else t for t in r["tags"]]
        merged.append({**r, "tags": list(dict.fromkeys(tags))})  # dedupe, keep order
    return merged


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rows = load_or_fetch()

    # Keep only articles the Guardian tagged as midterms coverage
    rows = [r for r in rows if "US midterm elections 2026" in r["tags"]]
    print(f"Filtered to {len(rows)} articles tagged as midterms")

    #updated merge of congress tags into one for Run D; comment out for Runs A-C
    rows = [r for r in rows if "US midterm elections 2026" in r["tags"]]
    rows = merge_congress(rows)   # Run D only; comment out for Runs A-C

    inspect(rows)
    G, pairs = build_graph(rows)
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"density {nx.density(G):.3f}")
    core, table = compute_centrality(G)
    top10 = pd.DataFrame({
        "Strength": table["strength"].sort_values(ascending=False).head(10).index,
        "Betweenness": table["betweenness"].sort_values(ascending=False).head(10).index,
        "Closeness": table["closeness"].sort_values(ascending=False).head(10).index,
    })
    top10.index = range(1, 11)
    top10.to_csv(OUT_DIR / "table1_top10.csv")
    print(top10)

    top_pairs(pairs)
    plot_network(core, table)
    plot_heatmap(pairs, table)
    plot_monthly(rows)
    
    print("\nSensitivity check (Trump and parties removed):")
    sensitivity(rows)
    print("\nDone. See output/ and figures/.")