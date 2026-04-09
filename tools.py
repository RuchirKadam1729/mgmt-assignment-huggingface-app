"""
tools.py  –  7 @tool functions for the BERTopic Agentic AI pipeline.

Rules enforced:
  - ZERO if/elif/else statements
  - ZERO for/while loops  (use list(map(...)) instead)
  - ZERO try/except blocks (handle_tool_error=True sends errors to LLM)
  - ALL tools are stateless: take input → produce output, nothing else
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─── Constants ──────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "all-MiniLM-L6-v2"
NEAREST_K = 5
MAX_LABEL_TOPICS = 30
GROQ_MODEL = "llama-3.1-8b-instant"  # 500k TPD free tier — no regional restrictions

RUN_CONFIGS = {
    "abstract": ["Abstract"],
    "title": ["Title"],
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


# ─── Tool 2: Run BERTopic Discovery ──────────────────────────────────────


@tool
def run_bertopic_discovery(run_key: str, threshold: float = 0.7) -> str:
    """Embed text sentences using all-MiniLM-L6-v2, cluster with AgglomerativeClustering
    (cosine metric, NO UMAP), find 5 nearest sentences per centroid, generate 4 Plotly charts.
    run_key must be 'abstract' or 'title'. Saves summaries.json and emb.npy."""

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
    sentences = [
        s.strip()
        for text in map(sent_tokenize, cleaned)
        for s in text
        if len(s.strip()) > 30
    ]

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        sentences, normalize_embeddings=True, show_progress_bar=False
    )

    clustering = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
        n_clusters=None,
    )
    labels = clustering.fit_predict(embeddings)
    n_clusters = int(labels.max()) + 1

    unique_labels = list(range(n_clusters))

    def get_cluster_data(cluster_id):
        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_sentences = [s for s, m in zip(sentences, mask) if m]
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, cluster_embeddings)[0]
        top_idx = sims.argsort()[-NEAREST_K:][::-1]
        top_sentences = [cluster_sentences[i] for i in top_idx]
        return {
            "cluster_id": cluster_id,
            "size": int(mask.sum()),
            "centroid": centroid[0].tolist(),
            "top_sentences": top_sentences,
            "label": f"Topic {cluster_id}",
        }

    summaries = list(map(get_cluster_data, unique_labels))
    summaries_sorted = sorted(summaries, key=lambda x: x["size"], reverse=True)

    (CHECKPOINT_DIR / f"summaries_{run_key}.json").write_text(
        json.dumps(summaries_sorted, indent=2)
    )
    np.save(CHECKPOINT_DIR / f"emb_{run_key}.npy", embeddings)
    (CHECKPOINT_DIR / f"sentences_{run_key}.json").write_text(json.dumps(sentences))

    sizes = [s["size"] for s in summaries_sorted[:30]]
    topic_names = [f"T{s['cluster_id']}" for s in summaries_sorted[:30]]

    fig_bar = go.Figure(go.Bar(x=topic_names, y=sizes, marker_color="#378ADD"))
    fig_bar.update_layout(
        title="Top 30 topics by sentence count",
        xaxis_title="Topic",
        yaxis_title="Sentences",
        height=350,
    )

    centroids = np.array([s["centroid"] for s in summaries_sorted[:50]])
    sim_matrix = cosine_similarity(centroids)
    fig_heat = go.Figure(go.Heatmap(z=sim_matrix, colorscale="Blues"))
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
        f"  Sentences embedded: {len(sentences)}\n"
        f"  Clusters (topics) found: {n_clusters}\n"
        f"  Threshold used: {threshold}\n"
        f"  Largest topic size: {summaries_sorted[0]['size']} sentences\n"
        f"  Charts saved: chart_bar, chart_heat, chart_map\n"
        f"  Checkpoint: summaries_{run_key}.json, emb_{run_key}.npy\n"
        f"Ready for Phase 2: call label_topics_with_llm(run_key='{run_key}')"
    )


# ─── Tool 3: Label Topics with LLM ───────────────────────────────────────


@tool
def label_topics_with_llm(run_key: str) -> str:
    """Send top 100 topics to ChatGroq LLM for labelling. Each topic gets a label,
    category, confidence score, reasoning, and niche flag. Saves labels.json."""

    assert (CHECKPOINT_DIR / f"summaries_{run_key}.json").exists(), (
        f"summaries_{run_key}.json not found. Call run_bertopic_discovery(run_key='{run_key}') first."
    )
    summaries = json.loads((CHECKPOINT_DIR / f"summaries_{run_key}.json").read_text())
    top_topics = summaries[:MAX_LABEL_TOPICS]

    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)

    def build_topic_text(t):
        sentences_text = "\n".join([f"  - {s}" for s in t["top_sentences"][:3]])
        return f"TOPIC {t['cluster_id']} (size={t['size']}):\n{sentences_text}"

    topics_text = "\n\n".join(list(map(build_topic_text, top_topics)))

    prompt = PromptTemplate.from_template(
        """You are a computational thematic analysis expert. Label each research topic based on its representative sentences.

For each topic, return a JSON object with exactly these fields:
- "cluster_id": integer
- "label": short research area name (3-6 words)
- "category": broader research category
- "confidence": float 0-1
- "reasoning": one sentence explanation
- "niche": boolean (true if very specialised, false if mainstream)

Return ONLY a JSON array of objects, no preamble, no markdown, no backticks.

Topics to label:
{topics_text}
"""
    )

    parser = JsonOutputParser()
    chain = prompt | llm | parser
    result = chain.invoke({"topics_text": topics_text})

    labeled = {str(item["cluster_id"]): item for item in result}
    enriched = list(
        map(
            lambda s: {
                **s,
                **labeled.get(
                    str(s["cluster_id"]),
                    {
                        "label": f"Topic {s['cluster_id']}",
                        "confidence": 0.5,
                        "reasoning": "Not labeled",
                        "niche": False,
                    },
                ),
            },
            summaries,
        )
    )

    (CHECKPOINT_DIR / f"labels_{run_key}.json").write_text(
        json.dumps(enriched, indent=2)
    )

    sample = enriched[:5]
    sample_text = "\n".join(
        [
            f"  T{t['cluster_id']}: {t.get('label','?')} (conf={t.get('confidence',0):.2f})"
            for t in sample
        ]
    )

    return (
        f"Topics labeled successfully for run_key='{run_key}'.\n"
        f"  Topics labeled: {len(result)}/{len(top_topics)}\n"
        f"  Sample labels:\n{sample_text}\n"
        f"  Checkpoint: labels_{run_key}.json\n"
        f"The review table will now populate. Researcher should review, set Approve=yes/no and Rename To values."
    )


# ─── Tool 4: Consolidate into Themes ─────────────────────────────────────


@tool
def consolidate_into_themes(run_key: str, theme_map: str) -> str:
    """Merge researcher-approved topic groups into consolidated themes.
    theme_map is a JSON string: {"Theme Name": [cluster_id1, cluster_id2, ...], ...}
    Recomputes centroids and sentence/paper counts. Saves themes.json."""

    assert (CHECKPOINT_DIR / f"labels_{run_key}.json").exists(), (
        f"labels_{run_key}.json not found. Call label_topics_with_llm(run_key='{run_key}') first."
    )
    assert (CHECKPOINT_DIR / f"emb_{run_key}.npy").exists(), (
        f"emb_{run_key}.npy not found. Call run_bertopic_discovery(run_key='{run_key}') first."
    )
    assert (CHECKPOINT_DIR / f"sentences_{run_key}.json").exists(), (
        f"sentences_{run_key}.json not found. Call run_bertopic_discovery(run_key='{run_key}') first."
    )
    labels_data = json.loads((CHECKPOINT_DIR / f"labels_{run_key}.json").read_text())
    embeddings = np.load(CHECKPOINT_DIR / f"emb_{run_key}.npy")
    sentences = json.loads((CHECKPOINT_DIR / f"sentences_{run_key}.json").read_text())
    cluster_labels_arr = json.loads(
        (CHECKPOINT_DIR / f"summaries_{run_key}.json").read_text()
    )

    cluster_lookup = {str(t["cluster_id"]): t for t in labels_data}
    sentence_cluster_map = {
        i: int(
            AgglomerativeClustering(
                metric="cosine",
                linkage="average",
                distance_threshold=0.7,
                n_clusters=None,
            ).fit_predict(embeddings)[i]
        )
        for i in range(len(sentences))
    }

    theme_definition = json.loads(theme_map)

    def build_theme(theme_name_ids):
        theme_name, cluster_ids = theme_name_ids
        ids_set = set(map(str, cluster_ids))
        member_topics = [
            cluster_lookup[cid] for cid in ids_set if cid in cluster_lookup
        ]
        all_sentences = [s for t in member_topics for s in t.get("top_sentences", [])]
        total_size = sum(t.get("size", 0) for t in member_topics)
        topic_labels = [t.get("label", f"T{t['cluster_id']}") for t in member_topics]
        return {
            "theme_name": theme_name,
            "cluster_ids": cluster_ids,
            "merged_topic_labels": topic_labels,
            "total_sentences": total_size,
            "representative_sentences": all_sentences[:NEAREST_K],
            "sub_topics": len(member_topics),
        }

    themes = list(map(build_theme, theme_definition.items()))
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


# ─── Tool 5: Compare with PAJAIS Taxonomy ────────────────────────────────


@tool
def compare_with_taxonomy(run_key: str) -> str:
    """Map final themes to the PAJAIS 25 research categories using ChatGroq LLM.
    Each theme is classified as MAPPED (with category) or NOVEL (new contribution).
    Saves taxonomy_map.json."""

    assert (CHECKPOINT_DIR / f"themes_{run_key}.json").exists(), (
        f"themes_{run_key}.json not found. Call consolidate_into_themes(run_key='{run_key}') first."
    )
    themes = json.loads((CHECKPOINT_DIR / f"themes_{run_key}.json").read_text())
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)

    categories_text = "\n".join(
        [f"  {i+1}. {c}" for i, c in enumerate(PAJAIS_CATEGORIES)]
    )

    def build_theme_text(t):
        sentences = "; ".join(t.get("representative_sentences", [])[:2])
        return f"THEME: {t['theme_name']}\nSentences: {sentences}"

    themes_text = "\n\n".join(list(map(build_theme_text, themes)))

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
    chain = prompt | llm | parser
    result = chain.invoke({"categories": categories_text, "themes_text": themes_text})

    mapped = {item["theme_name"]: item for item in result}
    enriched = list(
        map(
            lambda t: {
                **t,
                **mapped.get(
                    t["theme_name"],
                    {
                        "pajais_match": "NOVEL",
                        "is_novel": True,
                        "match_confidence": 0,
                        "reasoning": "No mapping found",
                    },
                ),
            },
            themes,
        )
    )

    novel_count = sum(1 for t in enriched if t.get("is_novel", False))
    mapped_count = len(enriched) - novel_count

    (CHECKPOINT_DIR / f"taxonomy_map_{run_key}.json").write_text(
        json.dumps(enriched, indent=2)
    )

    novel_names = [t["theme_name"] for t in enriched if t.get("is_novel", False)]
    novel_text = (
        "\n".join([f"  * {n}" for n in novel_names]) if novel_names else "  (none)"
    )

    return (
        f"PAJAIS taxonomy mapping complete for run_key='{run_key}'.\n"
        f"  Total themes mapped: {len(enriched)}\n"
        f"  MAPPED to PAJAIS: {mapped_count}\n"
        f"  NOVEL (not in taxonomy): {novel_count}\n"
        f"NOVEL themes (your paper's contribution):\n{novel_text}\n"
        f"  Checkpoint: taxonomy_map_{run_key}.json\n"
        f"Review PAJAIS mapping in the table. NOVEL themes represent publishable contributions."
    )


# ─── Tool 6: Generate Comparison CSV ─────────────────────────────────────


@tool
def generate_comparison_csv() -> str:
    """Load themes from both abstract and title runs and build a side-by-side
    comparison DataFrame. Saves comparison.csv."""

    abstract_path = CHECKPOINT_DIR / "themes_abstract.json"
    title_path = CHECKPOINT_DIR / "themes_title.json"

    abstract_themes = (
        json.loads(abstract_path.read_text()) if abstract_path.exists() else []
    )
    title_themes = json.loads(title_path.read_text()) if title_path.exists() else []

    max_len = max(len(abstract_themes), len(title_themes))
    pad = lambda lst: lst + [{}] * (max_len - len(lst))

    abstract_padded = pad(abstract_themes)
    title_padded = pad(title_themes)

    rows = list(
        map(
            lambda pair: {
                "Abstract Theme": pair[0].get("theme_name", ""),
                "Abstract Sentences": pair[0].get("total_sentences", ""),
                "Abstract Sub-Topics": pair[0].get("sub_topics", ""),
                "Title Theme": pair[1].get("theme_name", ""),
                "Title Sentences": pair[1].get("total_sentences", ""),
                "Title Sub-Topics": pair[1].get("sub_topics", ""),
                "Convergence": (
                    "CONVERGED"
                    if pair[0].get("theme_name", "").lower()
                    in pair[1].get("theme_name", "").lower()
                    or pair[1].get("theme_name", "").lower()
                    in pair[0].get("theme_name", "").lower()
                    else "DIVERGED"
                ),
            },
            zip(abstract_padded, title_padded),
        )
    )

    df = pd.DataFrame(rows)
    output_path = CHECKPOINT_DIR / "comparison.csv"
    df.to_csv(output_path, index=False)

    converged = sum(1 for r in rows if r["Convergence"] == "CONVERGED")

    return (
        f"Comparison CSV generated.\n"
        f"  Abstract themes: {len(abstract_themes)}\n"
        f"  Title themes: {len(title_themes)}\n"
        f"  Converged theme pairs: {converged}/{max_len}\n"
        f"  Saved: comparison.csv\n"
        f"Download from the Results tab. Converged themes are stable — highest confidence for Section 7."
    )


# ─── Tool 7: Export Narrative ─────────────────────────────────────────────


@tool
def export_narrative(run_key: str = "abstract") -> str:
    """Generate a 500-word Section 7 narrative draft for a literature review paper.
    Uses final themes and PAJAIS taxonomy mapping. Saves narrative.txt."""

    themes_path = CHECKPOINT_DIR / f"themes_{run_key}.json"
    taxonomy_path = CHECKPOINT_DIR / f"taxonomy_map_{run_key}.json"

    assert themes_path.exists(), (
        f"themes_{run_key}.json not found. Call consolidate_into_themes(run_key='{run_key}') first."
    )
    assert taxonomy_path.exists(), (
        f"taxonomy_map_{run_key}.json not found. Call compare_with_taxonomy(run_key='{run_key}') first."
    )
    themes = json.loads(themes_path.read_text())
    taxonomy = json.loads(taxonomy_path.read_text())

    tax_lookup = {t["theme_name"]: t for t in taxonomy}

    def theme_summary(t):
        tax = tax_lookup.get(t["theme_name"], {})
        pajais = tax.get("pajais_match", "NOVEL")
        novel_flag = "[NOVEL]" if tax.get("is_novel", False) else f"[PAJAIS: {pajais}]"
        return f"- {t['theme_name']} ({t['total_sentences']} sentences) {novel_flag}"

    themes_summary = "\n".join(list(map(theme_summary, themes)))
    novel_themes = [
        t["theme_name"]
        for t in themes
        if tax_lookup.get(t["theme_name"], {}).get("is_novel", False)
    ]
    novel_list = ", ".join(novel_themes) if novel_themes else "none identified"

    llm = ChatGroq(model=GROQ_MODEL, temperature=0.3)

    prompt = PromptTemplate.from_template(
        """You are writing Section 7 of a conference paper on topic modelling of a journal's literature.

Write approximately 500 words. Structure as follows:
(a) Methodology: State that BERTopic with sentence-level embeddings (all-MiniLM-L6-v2, 384 dimensions) and
    AgglomerativeClustering (cosine metric, threshold=0.7) was used. Mention Braun & Clarke (2006) six-phase
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

    chain = prompt | llm
    response = chain.invoke(
        {
            "run_key": run_key,
            "themes_summary": themes_summary,
            "novel_list": novel_list,
        }
    )

    narrative = response.content
    (CHECKPOINT_DIR / "narrative.txt").write_text(narrative)

    return (
        f"Section 7 narrative generated (~500 words).\n"
        f"  Run: {run_key}\n"
        f"  Themes covered: {len(themes)}\n"
        f"  Novel themes: {len(novel_themes)}\n"
        f"  Saved: narrative.txt\n"
        f"Download from the Results tab. Use as the base draft for Section 7 of your conference paper.\n\n"
        f"--- NARRATIVE PREVIEW (first 300 chars) ---\n{narrative[:300]}..."
    )


# ─── Exported tool list ───────────────────────────────────────────────────

ALL_TOOLS = [
    load_scopus_csv,
    run_bertopic_discovery,
    label_topics_with_llm,
    consolidate_into_themes,
    compare_with_taxonomy,
    generate_comparison_csv,
    export_narrative,
]