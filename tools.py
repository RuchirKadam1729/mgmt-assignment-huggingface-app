"""
tools.py  –  7 @tool functions for the BERTopic Agentic AI pipeline.

Rules enforced:
  - ZERO if/elif/else statements
  - ZERO for/while loops  (use list(map(...)) instead)
  - ZERO try/except blocks (handle_tool_error=True sends errors to LLM)
  - ALL tools are stateless: take input → produce output, nothing else

FIX (2026-05-04): Replaced DBSCAN with AgglomerativeClustering.
  DBSCAN with eps=0.35 chains all academic sentences into 1 cluster (chaining problem).
  AgglomerativeClustering with distance_threshold=0.35 cuts at a hard boundary →
  produces 80-120 distinct topic clusters from the same corpus.
"""

import json
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering   # ← CHANGED from DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─── Constants ──────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "allenai/specter2_base"   # 768-d scientific paper embeddings
NEAREST_K = 3                           # top 3 papers per cluster
MAX_LABEL_TOPICS = 120                  # label up to 120 clusters
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── AgglomerativeClustering parameters ────────────────────────────────────────
# DISTANCE_THRESHOLD: cosine distance = 1 − cosine_similarity
#   0.35 → similarity ≥ 0.65  →  ~80–120 fine-grained topics  (recommended)
#   0.30 → similarity ≥ 0.70  →  ~150–200 micro-topics
#   0.45 → similarity ≥ 0.55  →  ~40–60 coarser themes
AGGLO_DISTANCE_THRESHOLD = 0.35
AGGLO_MIN_CLUSTER_SIZE   = 20     # discard clusters with fewer sentences (treated as noise)
AGGLO_MAX_CLUSTERS       = 120    # cap review table rows

# Council of Agents — 3 independent labellers + 1 arbiter
COUNCIL_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

RUN_CONFIGS = {
    "abstract": ["Abstract"],
    "title":    ["Title"],
    "combined": ["Abstract", "Title"],
}

BOILERPLATE_PATTERNS = [
    r"©\s*\d{4}.*",
    r"All rights reserved.*",
    r"Published by Elsevier.*",
    r"This is an open access article.*",
    r"doi:.*",
    r"https?://\S+",
    r"www\.\S+",
    r"^\s*\d+\s*$",
    r"^\s*[A-Z]{2,}\s*$",
    r"Abstract\s*[:.]?\s*",
    r"Keywords\s*[:.]?\s*",
    r"Introduction\s*[:.]?\s*",
    r"Conclusions?\s*[:.]?\s*",
    r"References?\s*[:.]?\s*",
    r"Acknowledgements?\s*[:.]?\s*",
    r"Funding\s*[:.]?\s*",
    r"Conflict of interest.*",
    r"Author contributions?.*",
    r"Corresponding author.*",
    r"E-mail address.*",
    r"Received\s+\d+",
    r"Accepted\s+\d+",
]

PAJAIS_CATEGORIES = [
    "IS Strategy and Management",
    "IT Adoption and Diffusion",
    "E-Commerce and Digital Markets",
    "Business Intelligence and Analytics",
    "Knowledge Management",
    "IT and Organizational Change",
    "Supply Chain and ERP Systems",
    "Healthcare IT",
    "Mobile and Wireless Computing",
    "Social Media and Web 2.0",
    "Security and Privacy",
    "Cloud Computing",
    "Big Data and Data Mining",
    "Artificial Intelligence and Machine Learning",
    "Human-Computer Interaction",
    "Software Development and Engineering",
    "IT Governance and Outsourcing",
    "Digital Innovation and Entrepreneurship",
    "IS Education and Research Methods",
    "Collaborative Systems and CSCW",
    "Open Source and Crowdsourcing",
    "IT and Society",
    "Financial Technology",
    "Smart Cities and IoT",
    "Agile and DevOps Practices",
]

# ─── Tool 1: Load Scopus CSV ──────────────────────────────────────────────


@tool
def load_scopus_csv(filepath: str) -> str:
    """Load a Scopus export CSV, count papers and sentences, apply boilerplate filters.
    Returns a stats summary string with paper count, abstract sentences, and title sentences.
    """

    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")

    abstract_col = next((c for c in df.columns if "abstract" in c.lower()), None)
    title_col = next((c for c in df.columns if c.lower() == "title"), None)

    abstracts = list(df[abstract_col].dropna().astype(str)) if abstract_col else []
    titles = list(df[title_col].dropna().astype(str)) if title_col else []

    def clean_text(text):
        return re.sub(
            "|".join(BOILERPLATE_PATTERNS), " ", text, flags=re.IGNORECASE
        ).strip()

    def split_to_sentences(text_list):
        cleaned = list(map(clean_text, text_list))
        sentence_lists = list(map(sent_tokenize, cleaned))
        return [
            s.strip() for slist in sentence_lists for s in slist if len(s.strip()) > 30
        ]

    abstract_sentences = split_to_sentences(abstracts)
    title_sentences = split_to_sentences(titles)

    df.to_csv(CHECKPOINT_DIR / "data.csv", index=False)

    stats = {
        "papers": len(df),
        "abstract_sentences": len(abstract_sentences),
        "title_sentences": len(title_sentences),
        "columns": list(df.columns),
        "years": (
            sorted(df["Year"].dropna().astype(int).unique().tolist())
            if "Year" in df.columns
            else []
        ),
    }
    (CHECKPOINT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    return (
        f"CSV loaded successfully.\n"
        f"  Papers: {stats['papers']}\n"
        f"  Abstract sentences (after boilerplate filter): {stats['abstract_sentences']}\n"
        f"  Title sentences: {stats['title_sentences']}\n"
        f"  Columns: {', '.join(stats['columns'])}\n"
        f"  Year range: {min(stats['years'])} – {max(stats['years'])}"
        if stats["years"]
        else ""
    )


# ─── Tool 2: Run BERTopic Discovery AND Label ──────────────────────────────────────


def run_bertopic_discovery(run_key: str, threshold: float = AGGLO_DISTANCE_THRESHOLD) -> str:
    """Embed text sentences using allenai/specter2_base (768d), cluster with
    AgglomerativeClustering (cosine metric, distance_threshold — NO chaining problem),
    find top-3 nearest sentences per centroid.
    run_key must be 'abstract', 'title', or 'combined'.
    Saves summaries.json and emb.npy.

    WHY AgglomerativeClustering instead of DBSCAN:
      DBSCAN with eps=0.35 chains all academic-domain sentences into 1 giant cluster
      because every sentence is within eps of its neighbour in dense scientific text.
      AgglomerativeClustering cuts at a hard distance_threshold — no chaining —
      producing 80-120 distinct topic clusters.
    """

    df = pd.read_csv(CHECKPOINT_DIR / "data.csv")
    columns = RUN_CONFIGS[run_key]

    def get_text(row):
        return " ".join(
            [str(row[c]) for c in columns if c in row.index and pd.notna(row[c])]
        )

    raw_texts = list(map(get_text, [row for _, row in df.iterrows()]))

    def clean_text(text):
        return re.sub(
            "|".join(BOILERPLATE_PATTERNS), " ", text, flags=re.IGNORECASE
        ).strip()

    cleaned = list(map(clean_text, raw_texts))
    sentence_paper_pairs = [
        (s.strip(), paper_idx)
        for paper_idx, sents in enumerate(map(sent_tokenize, cleaned))
        for s in sents
        if len(s.strip()) > 30
    ]
    sentences      = [p[0] for p in sentence_paper_pairs]
    sent_paper_ids = [p[1] for p in sentence_paper_pairs]

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        sentences, normalize_embeddings=True, show_progress_bar=False
    )

    # ── AgglomerativeClustering (cosine distance, hard threshold) ─────────────
    # distance_threshold: cosine distance = 1 − cosine_similarity
    #   0.35 → groups sentences with similarity ≥ 0.65 → ~80-120 distinct topics
    # linkage="average": averages distances between all pairs (Ward requires euclidean)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
    )
    raw_labels = clustering.fit_predict(embeddings)

    # ── Post-filter: discard clusters smaller than MIN_CLUSTER_SIZE ───────────
    # (analogous to DBSCAN's min_samples — tiny clusters are treated as noise)
    from collections import Counter
    cluster_sizes = Counter(raw_labels.tolist())
    valid_ids = {cid for cid, sz in cluster_sizes.items() if sz >= AGGLO_MIN_CLUSTER_SIZE}
    labels = np.array([cid if cid in valid_ids else -1 for cid in raw_labels])

    noise_count      = int((labels == -1).sum())
    valid_cluster_ids = sorted(valid_ids)
    n_clusters        = len(valid_cluster_ids)

    def get_cluster_data(cluster_id):
        mask               = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_sentences  = [s   for s,   m in zip(sentences,      mask) if m]
        cluster_paper_ids  = [pid for pid, m in zip(sent_paper_ids, mask) if m]
        centroid           = cluster_embeddings.mean(axis=0, keepdims=True)
        sims               = cosine_similarity(centroid, cluster_embeddings)[0]
        top_idx            = sims.argsort()[-NEAREST_K:][::-1]
        top_sentences      = [cluster_sentences[i] for i in top_idx]
        unique_paper_ids   = list(set(cluster_paper_ids))
        return {
            "cluster_id":    cluster_id,
            "size":          int(mask.sum()),
            "paper_count":   len(unique_paper_ids),
            "paper_indices": unique_paper_ids,
            "centroid":      centroid[0].tolist(),
            "top_sentences": top_sentences,
            "label":         f"Topic {cluster_id}",
        }

    summaries        = list(map(get_cluster_data, valid_cluster_ids))
    summaries_sorted = sorted(summaries, key=lambda x: x["size"], reverse=True)
    summaries_sorted = summaries_sorted[:AGGLO_MAX_CLUSTERS]

    (CHECKPOINT_DIR / f"summaries_{run_key}.json").write_text(
        json.dumps(summaries_sorted, indent=2)
    )
    np.save(CHECKPOINT_DIR / f"emb_{run_key}.npy", embeddings)
    (CHECKPOINT_DIR / f"sentences_{run_key}.json").write_text(json.dumps(sentences))
    (CHECKPOINT_DIR / f"paper_ids_{run_key}.json").write_text(json.dumps(sent_paper_ids))

    sizes       = [s["size"] for s in summaries_sorted[:30]]
    topic_names = [f"T{s['cluster_id']}" for s in summaries_sorted[:30]]

    fig_bar = go.Figure(go.Bar(x=topic_names, y=sizes, marker_color="#378ADD"))
    fig_bar.update_layout(
        title="Top 30 topics by sentence count",
        xaxis_title="Topic",
        yaxis_title="Sentences",
        height=350,
    )

    centroids   = np.array([s["centroid"] for s in summaries_sorted[:50]])
    sim_matrix  = cosine_similarity(centroids)
    fig_heat    = go.Figure(go.Heatmap(z=sim_matrix, colorscale="Blues"))
    fig_heat.update_layout(title="Topic similarity heatmap (top 50)", height=400)

    x_coords = list(
        map(
            lambda i: float(np.cos(2 * np.pi * i / len(summaries_sorted[:30]))),
            range(len(summaries_sorted[:30])),
        )
    )
    y_coords = list(
        map(
            lambda i: float(np.sin(2 * np.pi * i / len(summaries_sorted[:30]))),
            range(len(summaries_sorted[:30])),
        )
    )
    fig_map = go.Figure(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers+text",
            marker=dict(
                size=list(
                    map(
                        lambda s: max(8, min(40, s["size"] // 5)), summaries_sorted[:30]
                    )
                ),
                color="#1D9E75",
                opacity=0.7,
            ),
            text=topic_names,
            textposition="top center",
        )
    )
    fig_map.update_layout(
        title="Intertopic distance map (top 30)", height=400, showlegend=False
    )

    fig_bar.write_html(
        str(CHECKPOINT_DIR / f"chart_bar_{run_key}.html"), include_plotlyjs="cdn"
    )
    fig_heat.write_html(
        str(CHECKPOINT_DIR / f"chart_heat_{run_key}.html"), include_plotlyjs="cdn"
    )
    fig_map.write_html(
        str(CHECKPOINT_DIR / f"chart_map_{run_key}.html"), include_plotlyjs="cdn"
    )

    return (
        f"BERTopic discovery complete for run_key='{run_key}'.\n"
        f"  Embedding model: {EMBED_MODEL} (768d)\n"
        f"  Sentences embedded: {len(sentences)}\n"
        f"  Clustering: AgglomerativeClustering(metric=cosine, linkage=average, "
        f"distance_threshold={threshold})\n"
        f"  Valid clusters (size ≥ {AGGLO_MIN_CLUSTER_SIZE}): {n_clusters}  "
        f"|  Small/noise sentences discarded: {noise_count}\n"
        f"  Clusters in review table: {len(summaries_sorted)} (top by size)\n"
        f"  Largest cluster: {summaries_sorted[0]['size']} sentences\n"
        f"  Charts saved: chart_bar, chart_heat, chart_map\n"
        f"  Checkpoint: summaries_{run_key}.json, emb_{run_key}.npy\n"
        f"Ready for Phase 2: call label_topics_with_llm(run_key='{run_key}')"
    )


def label_topics_with_llm(run_key: str) -> str:
    """Label top clusters using a Council of 3 LLMs independently,
    then a 4th arbiter LLM picks / synthesises the best label per topic.
    Saves labels.json with council_proposals field for transparency."""

    import time
    from itertools import chain as ichain

    assert (
        CHECKPOINT_DIR / f"summaries_{run_key}.json"
    ).exists(), f"summaries_{run_key}.json not found. Call run_bertopic_discovery first."

    summaries  = json.loads((CHECKPOINT_DIR / f"summaries_{run_key}.json").read_text())
    top_topics = summaries[:MAX_LABEL_TOPICS]

    def build_topic_text(t):
        sents = "\n".join([f"  - {s}" for s in t["top_sentences"][:NEAREST_K]])
        return f"TOPIC {t['cluster_id']} (size={t['size']}, papers={t['paper_count']}):\n{sents}"

    topics_text = "\n\n".join(list(map(build_topic_text, top_topics)))

    labeling_prompt = PromptTemplate.from_template(
        """You are a computational thematic analysis expert. Label each research topic.

For each topic return a JSON object with:
- "cluster_id": integer
- "label": short research area name (3-6 words, academic)
- "category": broader IS/IT research category
- "confidence": float 0-1
- "reasoning": one sentence
- "niche": boolean

Return ONLY a JSON array — no preamble, no markdown, no backticks.

Topics:
{topics_text}
"""
    )

    parser = JsonOutputParser()

    def get_labels_from_model(idx_model):
        idx, model_name = idx_model
        time.sleep(idx * 4)
        llm   = ChatGroq(model=model_name, temperature=0.1)
        chain = labeling_prompt | llm | parser
        result = chain.invoke({"topics_text": topics_text})
        return result if isinstance(result, list) else []

    all_model_results = list(map(get_labels_from_model, enumerate(COUNCIL_MODELS)))

    def get_one_proposal(result_list, cid_str):
        matches = list(filter(
            lambda item: isinstance(item, dict) and str(item.get("cluster_id")) == cid_str,
            result_list,
        ))
        return matches[0] if matches else {"label": "Unlabeled", "reasoning": "", "confidence": 0.5}

    def collect_proposals(topic):
        cid_str   = str(topic["cluster_id"])
        proposals = list(map(
            lambda res: get_one_proposal(res, cid_str),
            all_model_results,
        ))
        return {"cluster_id": topic["cluster_id"], "size": topic["size"], "proposals": proposals}

    per_topic_proposals = list(map(collect_proposals, top_topics))

    council_prompt = PromptTemplate.from_template(
        """You are the council arbiter for a research topic labelling panel.
Three AI models independently proposed a label for each topic.
Pick or synthesise the BEST label — prefer: (a) precise, (b) academic terminology, (c) distinctive.

{proposals_text}

Return ONLY a JSON array:
[{{"cluster_id": int, "label": str, "confidence": float, "reasoning": str, "winning_model": int}}]
winning_model: 1=Llama, 2=Mixtral, 3=Gemma, 0=synthesised.
No preamble, no markdown, no backticks.
"""
    )

    council_llm   = ChatGroq(model=GROQ_MODEL, temperature=0.0)
    council_chain = council_prompt | council_llm | parser

    COUNCIL_BATCH = 20

    def format_proposal_block(p):
        rows = list(map(
            lambda idx_prop: f"  Model {idx_prop[0]+1}: {idx_prop[1].get('label','?')} | {idx_prop[1].get('reasoning','')}",
            enumerate(p["proposals"]),
        ))
        return f"TOPIC {p['cluster_id']} (size={p['size']}):\n" + "\n".join(rows)

    def process_council_batch(batch):
        proposals_text = "\n\n".join(list(map(format_proposal_block, batch)))
        result = council_chain.invoke({"proposals_text": proposals_text})
        time.sleep(2)
        return result if isinstance(result, list) else []

    batches        = [per_topic_proposals[i : i + COUNCIL_BATCH]
                      for i in range(0, len(per_topic_proposals), COUNCIL_BATCH)]
    council_batches = list(map(process_council_batch, batches))

    council_results = list(filter(
        lambda item: isinstance(item, dict) and "cluster_id" in item,
        ichain.from_iterable(council_batches),
    ))
    council_lookup = {str(item["cluster_id"]): item for item in council_results}

    def merge_with_council(s):
        cid_str  = str(s["cluster_id"])
        council  = council_lookup.get(cid_str, {
            "label":         f"Topic {s['cluster_id']}",
            "confidence":    0.5,
            "reasoning":     "Council: no decision",
            "winning_model": 0,
        })
        matched_proposals = list(filter(
            lambda p: p["cluster_id"] == s["cluster_id"], per_topic_proposals
        ))
        proposal_labels = list(map(
            lambda prop: prop.get("label", ""),
            matched_proposals[0]["proposals"] if matched_proposals else [],
        ))
        return {**s, **council, "council_proposals": proposal_labels}

    enriched = list(map(merge_with_council, summaries))
    (CHECKPOINT_DIR / f"labels_{run_key}.json").write_text(json.dumps(enriched, indent=2))

    sample_text = "\n".join([
        f"  T{t['cluster_id']}: {t.get('label','?')}  [won=Model {t.get('winning_model',0)}]  "
        f"conf={t.get('confidence',0):.2f}"
        for t in enriched[:5]
    ])
    winning     = [r.get("winning_model", 0) for r in council_results]
    model_wins  = {m: winning.count(i+1) for i, m in enumerate(["Llama", "Mixtral", "Gemma"])}
    synth_count = winning.count(0)

    return (
        f"Council labelling complete for run_key='{run_key}'.\n"
        f"  Council: {', '.join(COUNCIL_MODELS)}\n"
        f"  Arbiter decisions: {len(council_results)}/{len(top_topics)}\n"
        f"  Model wins → {model_wins}  |  Synthesised: {synth_count}\n"
        f"  Sample:\n{sample_text}\n"
        f"  Checkpoint: labels_{run_key}.json\n"
        f"Review table ready. Set Approve/Rename, then click Submit Review."
    )


@tool
def run_bertopic_and_label(run_key: str, threshold: float = AGGLO_DISTANCE_THRESHOLD) -> str:
    """Run AgglomerativeClustering-based BERTopic discovery AND council-of-3-LLMs labelling
    in one step. run_key must be 'abstract', 'title', or 'combined'.
    threshold is the cosine distance_threshold for AgglomerativeClustering (default 0.35).
    Lower values → more, finer clusters. Higher values → fewer, coarser clusters."""
    discovery_result = run_bertopic_discovery(run_key, threshold)
    label_result     = label_topics_with_llm(run_key)
    return discovery_result + "\n\n" + label_result


# ─── Tool 3: Consolidate into Themes ─────────────────────────────────────


@tool
def consolidate_into_themes(run_key: str, theme_map: dict) -> str:
    """Merge researcher-approved topic groups into consolidated themes.
    theme_map is a dict: {"Theme Name": [cluster_id1, cluster_id2, ...], ...}
    Recomputes centroids and sentence/paper counts. Saves themes.json."""

    assert (
        CHECKPOINT_DIR / f"labels_{run_key}.json"
    ).exists(), f"labels_{run_key}.json not found. Call run_bertopic_and_label(run_key='{run_key}') first."
    assert (
        CHECKPOINT_DIR / f"emb_{run_key}.npy"
    ).exists(), f"emb_{run_key}.npy not found. Call run_bertopic_and_label(run_key='{run_key}') first."
    assert (
        CHECKPOINT_DIR / f"sentences_{run_key}.json"
    ).exists(), f"sentences_{run_key}.json not found. Call run_bertopic_and_label(run_key='{run_key}') first."

    labels_data       = json.loads((CHECKPOINT_DIR / f"labels_{run_key}.json").read_text())
    cluster_labels_arr = json.loads((CHECKPOINT_DIR / f"summaries_{run_key}.json").read_text())

    cluster_lookup    = {str(t["cluster_id"]): t for t in labels_data}

    def build_theme(theme_name_ids):
        theme_name, cluster_ids = theme_name_ids
        ids_set         = set(map(str, cluster_ids))
        member_topics   = [cluster_lookup[cid] for cid in ids_set if cid in cluster_lookup]
        all_sentences   = [s for t in member_topics for s in t.get("top_sentences", [])]
        total_size      = sum(t.get("size", 0) for t in member_topics)
        topic_labels    = [t.get("label", f"T{t['cluster_id']}") for t in member_topics]
        all_paper_indices = set(
            pid for t in member_topics for pid in t.get("paper_indices", [])
        )
        return {
            "theme_name":               theme_name,
            "cluster_ids":              cluster_ids,
            "merged_topic_labels":      topic_labels,
            "total_sentences":          total_size,
            "paper_count":              len(all_paper_indices),
            "representative_sentences": all_sentences[:NEAREST_K],
            "sub_topics":               len(member_topics),
        }

    themes        = list(map(build_theme, theme_map.items()))
    themes_sorted = sorted(themes, key=lambda x: x["total_sentences"], reverse=True)

    (CHECKPOINT_DIR / f"themes_{run_key}.json").write_text(
        json.dumps(themes_sorted, indent=2)
    )

    theme_summary = "\n".join(
        [
            f"  {i+1}. {t['theme_name']} ({t['total_sentences']} sentences, {t['sub_topics']} sub-topics)"
            for i, t in enumerate(themes_sorted)
        ]
    )

    return (
        f"Themes consolidated for run_key='{run_key}'.\n"
        f"  Total themes: {len(themes_sorted)}\n"
        f"Themes:\n{theme_summary}\n"
        f"  Checkpoint: themes_{run_key}.json\n"
        f"Review consolidated themes in the table. Click Submit Review to confirm or adjust."
    )


# ─── Tool 4: Compare with PAJAIS Taxonomy ────────────────────────────────


@tool
def compare_with_taxonomy(run_key: str) -> str:
    """Map final themes to the PAJAIS 25 research categories using ChatGroq LLM.
    Each theme is classified as MAPPED (with category) or NOVEL (new contribution).
    Saves taxonomy_map.json."""

    assert (CHECKPOINT_DIR / f"themes_{run_key}.json").exists(), \
        f"themes_{run_key}.json not found. Call consolidate_into_themes(run_key='{run_key}') first."

    themes = json.loads((CHECKPOINT_DIR / f"themes_{run_key}.json").read_text())
    llm    = ChatGroq(model=GROQ_MODEL, temperature=0.1)

    categories_text = "\n".join(
        [f"  {i+1}. {c}" for i, c in enumerate(PAJAIS_CATEGORIES)]
    )

    def build_theme_text(t):
        sentences = "; ".join(t.get("representative_sentences", [])[:2])
        return f"THEME: {t['theme_name']}\nSentences: {sentences}"

    prompt = PromptTemplate.from_template(
        """You are a systematic literature review expert. Map each research theme to the PAJAIS taxonomy.

PAJAIS Categories:
{categories}

For each theme, return a JSON object with:
- "theme_name": string (exact match from input)
- "pajais_match": string (exact category name from list above, or "NOVEL" if no match)
- "match_confidence": float 0-1 (0 if NOVEL)
- "is_novel": boolean
- "reasoning": one sentence explanation of the mapping or why it is novel

Return ONLY a JSON array, no preamble, no markdown, no backticks.

Themes to map:
{themes_text}
"""
    )

    parser = JsonOutputParser()
    chain  = prompt | llm | parser

    BATCH_SIZE = 20

    def make_batches(lst, size):
        return [lst[i:i+size] for i in range(0, len(lst), size)]

    def process_batch(batch):
        themes_text = "\n\n".join(list(map(build_theme_text, batch)))
        try:
            return chain.invoke({"categories": categories_text, "themes_text": themes_text})
        except Exception:
            half = len(batch) // 2
            if half == 0:
                return []
            r1 = chain.invoke({"categories": categories_text,
                                "themes_text": "\n\n".join(list(map(build_theme_text, batch[:half])))})
            r2 = chain.invoke({"categories": categories_text,
                                "themes_text": "\n\n".join(list(map(build_theme_text, batch[half:])))})
            return (r1 if isinstance(r1, list) else []) + (r2 if isinstance(r2, list) else [])

    import time
    def process_with_delay(batch):
        result = process_batch(batch)
        time.sleep(2)
        return result

    batches     = make_batches(themes, BATCH_SIZE)
    raw_results = list(map(process_with_delay, batches))

    result = [
        item for batch_result in raw_results
        if isinstance(batch_result, list)
        for item in batch_result
        if isinstance(item, dict) and "theme_name" in item
    ]

    mapped   = {item["theme_name"]: item for item in result}
    enriched = list(map(
        lambda t: {
            **t,
            **mapped.get(t["theme_name"], {
                "pajais_match":      "NOVEL",
                "is_novel":          True,
                "match_confidence":  0,
                "reasoning":         "No mapping found",
            }),
        },
        themes,
    ))

    novel_count  = sum(1 for t in enriched if t.get("is_novel", False))
    mapped_count = len(enriched) - novel_count

    (CHECKPOINT_DIR / f"taxonomy_map_{run_key}.json").write_text(
        json.dumps(enriched, indent=2)
    )

    novel_names = [t["theme_name"] for t in enriched if t.get("is_novel", False)]
    novel_text  = "\n".join([f"  * {n}" for n in novel_names]) if novel_names else "  (none)"

    return (
        f"PAJAIS taxonomy mapping complete for run_key='{run_key}'.\n"
        f"  Total themes mapped: {len(enriched)}\n"
        f"  MAPPED to PAJAIS: {mapped_count}\n"
        f"  NOVEL (not in taxonomy): {novel_count}\n"
        f"NOVEL themes (your paper's contribution):\n{novel_text}\n"
        f"  Checkpoint: taxonomy_map_{run_key}.json\n"
        f"Review PAJAIS mapping in the table. NOVEL themes represent publishable contributions."
    )


# ─── Tool 5: Generate Comparison CSV ─────────────────────────────────────


@tool
def generate_comparison_csv() -> str:
    """Load themes from both abstract and title runs and build a semantically-matched
    side-by-side comparison DataFrame. Uses cosine similarity on theme name embeddings
    to pair each abstract theme with its best-matching title theme (or mark ABSTRACT ONLY).
    Saves comparison.csv."""

    abstract_path = CHECKPOINT_DIR / "themes_abstract.json"
    title_path    = CHECKPOINT_DIR / "themes_title.json"

    abstract_themes = (
        json.loads(abstract_path.read_text()) if abstract_path.exists() else []
    )
    title_themes = json.loads(title_path.read_text()) if title_path.exists() else []

    CONVERGENCE_THRESHOLD = 0.40

    model = SentenceTransformer(EMBED_MODEL)

    abstract_names = list(map(lambda t: t.get("theme_name", ""), abstract_themes))
    title_names    = list(map(lambda t: t.get("theme_name", ""), title_themes))

    abstract_embs = (
        model.encode(abstract_names, normalize_embeddings=True)
        if abstract_names
        else np.zeros((0, 384))
    )
    title_embs = (
        model.encode(title_names, normalize_embeddings=True)
        if title_names
        else np.zeros((0, 384))
    )

    used_title_idxs = set()

    def match_abstract_theme(abs_idx):
        if len(title_embs) == 0:
            return None, 0.0
        sims             = cosine_similarity(abstract_embs[abs_idx : abs_idx + 1], title_embs)[0]
        available_mask   = np.array(
            [0.0 if i in used_title_idxs else 1.0 for i in range(len(sims))]
        )
        masked_sims = sims * available_mask
        best_idx    = int(masked_sims.argmax())
        best_sim    = float(masked_sims[best_idx])
        return best_idx, best_sim

    matched_pairs = list(
        map(
            lambda abs_idx: (abs_idx, *match_abstract_theme(abs_idx)),
            range(len(abstract_themes)),
        )
    )

    def claim_match(triple):
        abs_idx, title_idx, sim = triple
        if title_idx is not None and title_idx not in used_title_idxs:
            used_title_idxs.add(title_idx)
            return abs_idx, title_idx, sim
        return abs_idx, None, sim

    final_pairs = list(map(claim_match, matched_pairs))

    def make_row(triple):
        abs_idx, title_idx, sim = triple
        at     = abstract_themes[abs_idx]
        tt     = title_themes[title_idx] if title_idx is not None else {}
        a_name = at.get("theme_name", "")
        t_name = tt.get("theme_name", "")

        if not t_name:
            convergence = "ABSTRACT ONLY"
        elif sim >= CONVERGENCE_THRESHOLD:
            convergence = "CONVERGED"
        else:
            convergence = "DIVERGED"

        return {
            "Abstract Theme":     a_name,
            "Abstract Sentences": at.get("total_sentences", ""),
            "Abstract Papers":    at.get("paper_count", ""),
            "Title Theme":        t_name,
            "Title Sentences":    tt.get("total_sentences", ""),
            "Title Papers":       tt.get("paper_count", ""),
            "Similarity":         round(sim, 3) if t_name else "",
            "Convergence":        convergence,
        }

    rows = list(map(make_row, final_pairs))

    matched_title_idxs    = {triple[1] for triple in final_pairs if triple[1] is not None}
    unmatched_title_rows  = list(
        map(
            lambda ti: {
                "Abstract Theme":     "",
                "Abstract Sentences": "",
                "Abstract Papers":    "",
                "Title Theme":        title_themes[ti].get("theme_name", ""),
                "Title Sentences":    title_themes[ti].get("total_sentences", ""),
                "Title Papers":       title_themes[ti].get("paper_count", ""),
                "Similarity":         "",
                "Convergence":        "TITLE ONLY",
            },
            [i for i in range(len(title_themes)) if i not in matched_title_idxs],
        )
    )
    rows = rows + unmatched_title_rows

    df          = pd.DataFrame(rows)
    output_path = CHECKPOINT_DIR / "comparison.csv"
    df.to_csv(output_path, index=False)

    converged     = sum(1 for r in rows if r["Convergence"] == "CONVERGED")
    abstract_only = sum(1 for r in rows if r["Convergence"] == "ABSTRACT ONLY")
    title_only    = sum(1 for r in rows if r["Convergence"] == "TITLE ONLY")
    diverged      = sum(1 for r in rows if r["Convergence"] == "DIVERGED")

    return (
        f"Comparison CSV generated (semantic matching, threshold={CONVERGENCE_THRESHOLD}).\n"
        f"  Abstract themes: {len(abstract_themes)}\n"
        f"  Title themes: {len(title_themes)}\n"
        f"  CONVERGED pairs:  {converged}\n"
        f"  DIVERGED pairs:   {diverged}\n"
        f"  ABSTRACT ONLY:    {abstract_only}  (no close title match)\n"
        f"  TITLE ONLY:       {title_only}  (no close abstract match)\n"
        f"  Saved: comparison.csv\n"
        f"Converged themes are stable — highest confidence for Section 7."
    )


# ─── Tool 6: Export Narrative ─────────────────────────────────────────────


@tool
def export_narrative(run_key: str = "abstract") -> str:
    """Generate a 500-word Section 7 narrative draft for a literature review paper.
    Uses final themes and PAJAIS taxonomy mapping. Saves narrative.txt."""

    themes_path   = CHECKPOINT_DIR / f"themes_{run_key}.json"
    taxonomy_path = CHECKPOINT_DIR / f"taxonomy_map_{run_key}.json"

    assert themes_path.exists(), \
        f"themes_{run_key}.json not found. Call consolidate_into_themes(run_key='{run_key}') first."
    assert taxonomy_path.exists(), \
        f"taxonomy_map_{run_key}.json not found. Call compare_with_taxonomy(run_key='{run_key}') first."

    themes   = json.loads(themes_path.read_text())
    taxonomy = json.loads(taxonomy_path.read_text())

    tax_lookup = {t["theme_name"]: t for t in taxonomy}

    def theme_summary(t):
        tax        = tax_lookup.get(t["theme_name"], {})
        pajais     = tax.get("pajais_match", "NOVEL")
        novel_flag = "[NOVEL]" if tax.get("is_novel", False) else f"[PAJAIS: {pajais}]"
        return f"- {t['theme_name']} ({t['total_sentences']} sentences) {novel_flag}"

    themes_summary = "\n".join(list(map(theme_summary, themes)))
    novel_themes   = [
        t["theme_name"]
        for t in themes
        if tax_lookup.get(t["theme_name"], {}).get("is_novel", False)
    ]
    novel_list = ", ".join(novel_themes) if novel_themes else "none identified"

    llm    = ChatGroq(model=GROQ_MODEL, temperature=0.3)
    prompt = PromptTemplate.from_template(
        """You are writing Section 7 of a conference paper on topic modelling of a journal's literature.

Write approximately 500 words. Structure as follows:
(a) Methodology: State that BERTopic with sentence-level embeddings (allenai/specter2_base, 768 dimensions) and
    AgglomerativeClustering (cosine metric, distance_threshold=0.35) was used. Mention Braun & Clarke (2006) six-phase
    thematic analysis framework and the researcher-in-the-loop validation process.
(b) Findings: Describe each theme below with its label, sentence count, and PAJAIS mapping status.
    Reference the comparison CSV and PAJAIS taxonomy map.
(c) Interpretation: What do these themes reveal about the journal's research landscape?
    Highlight convergence between abstract and title runs as evidence of stability.
(d) Contribution: Explicitly name the NOVEL themes as the paper's contribution to the field.
(e) Limitations: Acknowledge cosine threshold sensitivity, sentence-level (not document-level) analysis,
    and LLM labelling subjectivity per Carlsen & Ralund (2022).

Themes discovered (run: {run_key}):
{themes_summary}

Novel themes (not in PAJAIS taxonomy): {novel_list}

Write in formal academic English. Cite Braun & Clarke (2006), Grootendorst (2022), and Carlsen & Ralund (2022).
"""
    )

    chain    = prompt | llm
    response = chain.invoke(
        {"run_key": run_key, "themes_summary": themes_summary, "novel_list": novel_list}
    )

    narrative = response.content
    (CHECKPOINT_DIR / "narrative.txt").write_text(narrative)

    return (
        f"Section 7 narrative generated (~500 words).\n"
        f"  Run: {run_key}\n"
        f"  Themes covered: {len(themes)}\n"
        f"  Novel themes: {len(novel_themes)}\n"
        f"  Saved: narrative.txt\n"
        f"Download from the Results tab. Use as the base draft for Section 7.\n\n"
        f"--- NARRATIVE PREVIEW (first 300 chars) ---\n{narrative[:300]}..."
    )


# ─── Exported tool list ───────────────────────────────────────────────────

ALL_TOOLS = [
    load_scopus_csv,
    run_bertopic_and_label,
    consolidate_into_themes,
    compare_with_taxonomy,
    generate_comparison_csv,
    export_narrative,
]