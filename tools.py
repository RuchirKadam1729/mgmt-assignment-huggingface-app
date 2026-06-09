"""
tools.py  –  7 @tool functions for the BERTopic Agentic AI pipeline.

Rules enforced:
  - ZERO if/elif/else statements
  - ZERO for/while loops  (use list(map(...)) instead)
  - ALL tools are stateless: take input → produce output, nothing else

NOTE (2026-06-05): The "ZERO try/except" rule was revoked.
  handle_tool_errors=True only works when exceptions stay inside ToolNode.
  Rate-limit / JSON-parse exceptions from internal LLM calls escape before
  ToolNode can wrap them, writing a dangling AIMessage(tool_calls=[...]) to
  MemorySaver → thread permanently broken.  Every @tool now catches all
  exceptions and returns an error string so the checkpoint stays valid.

FIX (2026-05-04): Replaced DBSCAN with AgglomerativeClustering.
  DBSCAN with eps=1.5 chains all academic sentences into 1 cluster (chaining problem).
  AgglomerativeClustering with distance_threshold=1.5 cuts at a hard boundary →
  produces 80-120 distinct topic clusters from the same corpus.

FIX (2026-05-30): Batched per-model labeling calls (LABEL_BATCH_SIZE=15).
  Sending all 120 topics at once = ~15,872 tokens per call, which crashes Groq's
  6,000 TPM free-tier limit with a 429 error. 15 topics per batch = ~1,981 tokens,
  safely under the limit. Also added max_tokens=4096 to prevent truncated JSON.
  Replaced meta-llama/llama-4-scout-17b-16e-instruct with gemma2-9b-it (stable on
  Groq free tier, no model-name format issues).

FIX (2026-06-04): Replaced decommissioned gemma2-9b-it with qwen/qwen3-32b.
  gemma2-9b-it was shut down by Groq on 2025-10-08. llama-3.1-8b-instant (Groq's
  official replacement) is already council[1], so qwen/qwen3-32b is used instead
  to preserve architectural diversity in the 3-model council.
"""

import json
import re
import time
import traceback
from collections import Counter
from itertools import chain as ichain
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from nltk.tokenize import sent_tokenize
from pydantic import SecretStr
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─── Constants ──────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "allenai/specter2_base"  # 768-d scientific paper embeddings
NEAREST_K = 3  # top 3 sentences per cluster centroid
MAX_LABEL_TOPICS = 120  # label up to 120 clusters
# Agent coordination model — llama-3.1-8b-instant has 500K TPD vs 100K for 70b.
# The 70b model is kept as one voice in COUNCIL_MODELS for labelling quality,
# but using it for the agent brain + taxonomy + council simultaneously burns the
# 100K daily limit before Phase 2 even finishes.
GROQ_MODEL = "llama-3.1-8b-instant"

# ── AgglomerativeClustering parameters ────────────────────────────────────────
AGGLO_DISTANCE_THRESHOLD = 1.5
AGGLO_MIN_CLUSTER_SIZE = 20  # discard clusters smaller than this (noise)
AGGLO_MAX_CLUSTERS = 120  # cap review table rows

# ── Batch sizes for per-model labelling ──────────────────────────────────────
# Qwen3-32b is limited to 6000 TPM on Groq free tier; each topic prompt is
# ~400 tokens, so 15 × 400 = 6000 → hits the limit exactly (actually 6007 →
# 413 error on every batch).  Use 10 for Qwen3, 15 for the others.
LABEL_BATCH_SIZE = 15
LABEL_BATCH_SIZE_QWEN = 10  # Qwen3-32b: 6000 TPM ÷ ~500 tokens/topic ≈ 10

# Council of Agents — 3 independent labellers + 1 arbiter
COUNCIL_MODELS = [
    "llama-3.3-70b-versatile",  # primary — high quality
    "llama-3.1-8b-instant",     # fast lightweight voice
    "qwen/qwen3-32b",           # third voice — different architecture
]

RUN_CONFIGS = {
    "abstract": ["Abstract"],
    "title": ["Title"],
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

# ─── Robust JSON parser ──────────────────────────────────────────────────────
#
# FIX (2026-06-05): Replace JsonOutputParser with this everywhere.
# qwen/qwen3-32b emits <think>…</think> blocks before the JSON array on Groq.
# JsonOutputParser sees the thinking text and raises "Invalid json output".


def _make_json_parser() -> RunnableLambda:
    """Strip Qwen3 <think> blocks + markdown fences, then extract the first JSON array."""

    def _parse(msg):
        text = msg.content if hasattr(msg, "content") else str(msg)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        return json.loads(text)  # fallback: try the whole cleaned string

    return RunnableLambda(_parse)


# ─── Tool 1: Load Scopus CSV ──────────────────────────────────────────────


@tool
def load_scopus_csv(filepath: str) -> str:
    """Load a Scopus export CSV, count papers and sentences, apply boilerplate filters.
    Returns a stats summary string with paper count, abstract sentences, and title sentences.
    """
    try:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")

        abstract_col = next(filter(lambda c: "abstract" in c.lower(), df.columns), None)
        title_col = next(filter(lambda c: c.lower() == "title", df.columns), None)

        abstracts = list(df[abstract_col].dropna().astype(str)) if abstract_col else []
        titles = list(df[title_col].dropna().astype(str)) if title_col else []

        def clean_text(text):
            return re.sub(
                "|".join(BOILERPLATE_PATTERNS), " ", text, flags=re.IGNORECASE
            ).strip()

        def split_to_sentences(text_list):
            return list(
                filter(
                    lambda s: len(s) > 30,
                    map(
                        lambda s: s.strip(),
                        ichain.from_iterable(
                            map(sent_tokenize, map(clean_text, text_list))
                        ),
                    ),
                )
            )

        abstract_sentences = split_to_sentences(abstracts)
        title_sentences = split_to_sentences(titles)

        df.to_csv(CHECKPOINT_DIR / "data.csv", index=False)

        stats = {
            "papers": len(df),
            "abstract_sentences": len(abstract_sentences),
            "title_sentences": len(title_sentences),
            "columns": list(df.columns),
            "years": sorted(df["Year"].dropna().astype(int).unique().tolist())
            if "Year" in df.columns
            else [],
        }
        (CHECKPOINT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

        return (
            (
                f"CSV loaded successfully.\n"
                f"  Papers: {stats['papers']}\n"
                f"  Abstract sentences (after boilerplate filter): {stats['abstract_sentences']}\n"
                f"  Title sentences: {stats['title_sentences']}\n"
                f"  Columns: {', '.join(stats['columns'])}\n"
                f"  Year range: {min(stats['years'])} – {max(stats['years'])}"
            )
            if stats["years"]
            else "CSV loaded successfully, but no valid publishing years were identified."
        )
    except Exception as exc:
        return f"❌ load_scopus_csv failed: {type(exc).__name__}: {exc}"


# ─── Tool 2: BERTopic Discovery (helper, not @tool) ──────────────────────────


def run_bertopic_discovery(
    run_key: str, threshold: float = AGGLO_DISTANCE_THRESHOLD
) -> str:
    """Embed sentences using allenai/specter2_base (768d), reduce with UMAP to 10d,
    cluster with AgglomerativeClustering (Ward/Euclidean, hard threshold — no chaining).
    Saves summaries.json, emb.npy, and three Plotly chart HTML files."""
    df = pd.read_csv(CHECKPOINT_DIR / "data.csv")
    columns = RUN_CONFIGS[run_key]

    def get_text(row):
        return " ".join(
            map(
                lambda c: str(row[c]),
                filter(lambda c: c in row.index and pd.notna(row[c]), columns),
            )
        )

    raw_texts = list(map(get_text, map(lambda pair: pair[1], df.iterrows())))

    def clean_text(text):
        return re.sub(
            "|".join(BOILERPLATE_PATTERNS), " ", text, flags=re.IGNORECASE
        ).strip()

    cleaned = list(map(clean_text, raw_texts))

    def process_paper(pair):
        return map(lambda s: (s.strip(), pair[0]), pair[1])

    all_pairs = ichain.from_iterable(
        map(process_paper, enumerate(map(sent_tokenize, cleaned)))
    )
    sentence_paper_pairs = list(filter(lambda p: len(p[0]) > 30, all_pairs))

    sentences = list(map(lambda p: p[0], sentence_paper_pairs))
    sent_paper_ids = list(map(lambda p: p[1], sentence_paper_pairs))

    emb_path = CHECKPOINT_DIR / f"emb_{run_key}.npy"
    sentences_path = CHECKPOINT_DIR / f"sentences_{run_key}.json"
    paper_ids_path = CHECKPOINT_DIR / f"paper_ids_{run_key}.json"

    # ── Embedding cache ───────────────────────────────────────────────────────
    # Encoding + UMAP takes 5-10 minutes and burns thousands of tokens on each
    # retry.  Persist the embeddings after the first run so that interrupted
    # runs (rate-limit, crash, etc.) skip straight to re-clustering (seconds).
    if emb_path.exists() and sentences_path.exists():
        print(
            f"[BERTopic] Cache hit for run_key='{run_key}' — "
            "loading embeddings, skipping encode+UMAP."
        )
        embeddings = np.load(emb_path)
        sentences = json.loads(sentences_path.read_text())
        sent_paper_ids = (
            json.loads(paper_ids_path.read_text()) if paper_ids_path.exists() else []
        )
        reduced_embeddings = None  # will use embeddings directly for clustering
    else:
        model = SentenceTransformer(EMBED_MODEL)
        embeddings = model.encode(
            sentences, normalize_embeddings=True, show_progress_bar=False
        )
        np.save(emb_path, embeddings)
        sentences_path.write_text(json.dumps(sentences))
        paper_ids_path.write_text(json.dumps(sent_paper_ids))
        reduced_embeddings = None  # computed below

    if reduced_embeddings is None:
        reducer = UMAP(
            n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42
        )
        reduced_embeddings = reducer.fit_transform(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="euclidean",
        linkage="ward",
        distance_threshold=threshold,
    )
    raw_labels = clustering.fit_predict(reduced_embeddings)

    cluster_sizes = Counter(raw_labels.tolist())
    valid_ids = set(
        filter(lambda cid: cluster_sizes[cid] >= AGGLO_MIN_CLUSTER_SIZE, cluster_sizes)
    )
    labels = np.array(
        list(map(lambda cid: cid if cid in valid_ids else -1, raw_labels))
    )

    noise_count = int((labels == -1).sum())
    valid_cluster_ids = sorted(valid_ids)
    n_clusters = len(valid_cluster_ids)

    def get_cluster_data(cluster_id):
        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_sentences = list(
            map(lambda p: p[0], filter(lambda p: p[1], zip(sentences, mask)))
        )
        cluster_paper_ids = list(
            map(lambda p: p[0], filter(lambda p: p[1], zip(sent_paper_ids, mask)))
        )
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, cluster_embeddings)[0]
        top_idx = sims.argsort()[-NEAREST_K:][::-1]
        top_sentences = list(map(lambda i: cluster_sentences[i], top_idx))
        unique_paper_ids = list(set(cluster_paper_ids))
        return {
            "cluster_id": cluster_id,
            "size": int(mask.sum()),
            "paper_count": len(unique_paper_ids),
            "paper_indices": unique_paper_ids,
            "centroid": centroid[0].tolist(),
            "top_sentences": top_sentences,
            "label": f"Topic {cluster_id}",
        }

    summaries = list(map(get_cluster_data, valid_cluster_ids))
    summaries_sorted = sorted(summaries, key=lambda x: x["size"], reverse=True)[
        :AGGLO_MAX_CLUSTERS
    ]

    (CHECKPOINT_DIR / f"summaries_{run_key}.json").write_text(
        json.dumps(summaries_sorted, indent=2)
    )
    # emb, sentences, paper_ids already written in the cache block above

    sizes = list(map(lambda s: s["size"], summaries_sorted[:30]))
    topic_names = list(map(lambda s: f"T{s['cluster_id']}", summaries_sorted[:30]))

    fig_bar = go.Figure(go.Bar(x=topic_names, y=sizes, marker_color="#378ADD"))
    fig_bar.update_layout(
        title="Top 30 topics by sentence count",
        xaxis_title="Topic",
        yaxis_title="Sentences",
        height=350,
    )

    centroids = np.array(list(map(lambda s: s["centroid"], summaries_sorted[:50])))
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
        f"  Embedding model: {EMBED_MODEL} (768d) → UMAP 10d\n"
        f"  Sentences embedded: {len(sentences)}\n"
        f"  Clustering: AgglomerativeClustering(metric=euclidean, linkage=ward, distance_threshold={threshold})\n"
        f"  Valid clusters (size ≥ {AGGLO_MIN_CLUSTER_SIZE}): {n_clusters}  |  Small/noise sentences discarded: {noise_count}\n"
        f"  Clusters in review table: {len(summaries_sorted)} (top by size)\n"
        f"  Largest cluster: {summaries_sorted[0]['size'] if summaries_sorted else 0} sentences\n"
        f"  Charts saved: chart_bar, chart_heat, chart_map\n"
        f"  Checkpoint: summaries_{run_key}.json, emb_{run_key}.npy\n"
        f"Ready for labelling: call label_topics_with_llm(run_key='{run_key}')"
    )


# ─── Tool 2b: Council-of-3 LLM Labelling (helper, not @tool) ─────────────────


def label_topics_with_llm(run_key: str) -> str:
    """Label top clusters using a Council of 3 LLMs independently,
    then a 4th arbiter LLM picks / synthesises the best label per topic.
    Saves labels.json with council_proposals field for transparency.
    """
    assert (CHECKPOINT_DIR / f"summaries_{run_key}.json").exists(), (
        f"summaries_{run_key}.json not found. Run discovery first."
    )

    summaries = json.loads((CHECKPOINT_DIR / f"summaries_{run_key}.json").read_text())
    top_topics = summaries[:MAX_LABEL_TOPICS]

    def build_topic_text(t):
        return (
            f"TOPIC {t['cluster_id']} (size={t['size']}, papers={t['paper_count']}):\n"
            + "\n".join(map(lambda s: f"  - {s}", t["top_sentences"][:NEAREST_K]))
        )

    labeling_prompt = PromptTemplate.from_template(
        "You are a computational thematic analysis expert. Label each research topic.\n\n"
        "For each topic return a JSON object with:\n"
        '- "cluster_id": integer\n'
        '- "label": short research area name (3-6 words, academic)\n'
        '- "category": broader IS/IT research category\n'
        '- "confidence": float 0-1\n'
        '- "reasoning": one sentence\n'
        '- "niche": boolean\n\n'
        "Return ONLY a JSON array — no preamble, no markdown, no backticks.\n\n"
        "Topics:\n{topics_text}"
    )

    parser = _make_json_parser()  # FIX: handles Qwen3 <think> blocks

    def get_labels_from_model(idx_model):
        idx, model_name = idx_model
        time.sleep(idx * 6)  # stagger starts to prevent synchronized bursts

        from agent import KEY_ROTATOR

        # NOTE: thinking_budget_tokens was removed — Groq SDK does not accept it as a
        # model_kwarg. <think> blocks are stripped by _make_json_parser() instead.
        llm = ChatGroq(
            model=model_name,
            api_key=SecretStr(KEY_ROTATOR.next()),
            temperature=0.1,
            max_tokens=4096,
            max_retries=5,
        )
        chain = labeling_prompt | llm | parser

        # Qwen3-32b has a 6000 TPM limit — use a smaller batch so requests don't 413
        batch_size = LABEL_BATCH_SIZE_QWEN if "qwen" in model_name.lower() else LABEL_BATCH_SIZE

        label_batches = list(
            map(
                lambda i: top_topics[i : i + batch_size],
                range(0, len(top_topics), batch_size),
            )
        )

        def invoke_batch(batch):
            # FIX: catch per-batch failures so one bad batch doesn't kill the model
            try:
                batch_text = "\n\n".join(list(map(build_topic_text, batch)))
                res = chain.invoke({"topics_text": batch_text})
                time.sleep(4)  # throttle against TPM limits
                return res if isinstance(res, list) else []
            except Exception as exc:
                batch_ids = list(map(lambda t: t["cluster_id"], batch))
                print(
                    f"[LabelBatch] {model_name} failed on batch {batch_ids}: "
                    f"{type(exc).__name__}: {exc}"
                )
                return []  # graceful degradation: this batch gets fallback labels

        return list(ichain.from_iterable(map(invoke_batch, label_batches)))

    def get_labels_from_model_safe(idx_model):
        # FIX: if an entire model fails, return empty list rather than crashing
        try:
            return get_labels_from_model(idx_model)
        except Exception as exc:
            _, model_name = idx_model
            print(
                f"[CouncilModel] {model_name} failed entirely: "
                f"{type(exc).__name__}: {exc}"
            )
            return []

    all_model_results = list(map(get_labels_from_model_safe, enumerate(COUNCIL_MODELS)))

    def get_one_proposal(result_list, cid_str):
        return next(
            filter(
                lambda item: (
                    isinstance(item, dict) and str(item.get("cluster_id")) == cid_str
                ),
                result_list,
            ),
            {
                "label": "Unlabeled",
                "reasoning": "Fallback configuration assignment",
                "confidence": 0.5,
            },
        )

    def collect_proposals(topic):
        return {
            "cluster_id": topic["cluster_id"],
            "size": topic["size"],
            "proposals": list(
                map(
                    lambda res: get_one_proposal(res, str(topic["cluster_id"])),
                    all_model_results,
                )
            ),
        }

    per_topic_proposals = list(map(collect_proposals, top_topics))

    council_prompt = PromptTemplate.from_template(
        "You are the council arbiter for a research topic labelling panel.\n"
        "Three AI models independently proposed a label for each topic.\n"
        "Pick or synthesise the BEST label — prefer: (a) precise, (b) academic terminology, (c) distinctive.\n\n"
        "{proposals_text}\n\n"
        "Return ONLY a JSON array:\n"
        '[{{"cluster_id": int, "label": str, "confidence": float, "reasoning": str, "winning_model": int}}]\n'
        "winning_model: 1=Llama-70b, 2=Llama-8b, 3=Qwen3-32b, 0=synthesised.\n"
        "No preamble, no markdown, no backticks."
    )

    from agent import KEY_ROTATOR

    council_llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=SecretStr(KEY_ROTATOR.next()),
        temperature=0.0,
        max_tokens=4096,
        max_retries=5,
    )
    council_chain = council_prompt | council_llm | _make_json_parser()  # FIX: robust parser

    COUNCIL_BATCH = 20

    def format_proposal_block(p):
        return f"TOPIC {p['cluster_id']} (size={p['size']}):\n" + "\n".join(
            map(
                lambda idx_prop: (
                    f"  Model {idx_prop[0] + 1}: {idx_prop[1].get('label', '?')} | {idx_prop[1].get('reasoning', '')}"
                ),
                enumerate(p["proposals"]),
            )
        )

    def process_council_batch(batch):
        # FIX: catch per-batch failures — a bad batch returns empty, not a crash
        try:
            proposals_text = "\n\n".join(list(map(format_proposal_block, batch)))
            result = council_chain.invoke({"proposals_text": proposals_text})
            time.sleep(2)
            return result if isinstance(result, list) else []
        except Exception as exc:
            batch_ids = list(map(lambda p: p["cluster_id"], batch))
            print(
                f"[CouncilBatch] Arbiter failed on batch {batch_ids}: "
                f"{type(exc).__name__}: {exc}"
            )
            return []  # these topics will get fallback labels

    batches = list(
        map(
            lambda i: per_topic_proposals[i : i + COUNCIL_BATCH],
            range(0, len(per_topic_proposals), COUNCIL_BATCH),
        )
    )
    council_results = list(
        filter(
            lambda item: isinstance(item, dict) and "cluster_id" in item,
            ichain.from_iterable(map(process_council_batch, batches)),
        )
    )
    council_lookup = dict(
        map(lambda item: (str(item["cluster_id"]), item), council_results)
    )

    def merge_with_council(s):
        cid_str = str(s["cluster_id"])
        council = council_lookup.get(
            cid_str,
            {
                "label": f"Topic {s['cluster_id']}",
                "confidence": 0.5,
                "reasoning": "Council fallback mapping active",
                "winning_model": 0,
            },
        )
        matched_proposals = list(
            filter(lambda p: p["cluster_id"] == s["cluster_id"], per_topic_proposals)
        )
        proposal_labels = (
            list(
                map(
                    lambda prop: prop.get("label", ""),
                    matched_proposals[0]["proposals"],
                )
            )
            if matched_proposals
            else []
        )
        return {**s, **council, "council_proposals": proposal_labels}

    enriched = list(map(merge_with_council, summaries))
    (CHECKPOINT_DIR / f"labels_{run_key}.json").write_text(
        json.dumps(enriched, indent=2)
    )

    sample_text = "\n".join(
        map(
            lambda t: (
                f"  T{t['cluster_id']}: {t.get('label', '?')}  [won=Model {t.get('winning_model', 0)}]  conf={t.get('confidence', 0):.2f}"
            ),
            enriched[:5],
        )
    )
    winning = list(map(lambda r: r.get("winning_model", 0), council_results))
    model_wins = dict(
        map(
            lambda pair: (pair[1], winning.count(pair[0] + 1)),
            enumerate(["Llama-70b", "Llama-8b", "Gemma"]),
        )
    )
    synth_count = winning.count(0)

    return (
        f"Council labelling complete for run_key='{run_key}'.\n"
        f"  Council: {', '.join(COUNCIL_MODELS)}\n"
        f"  Batches per model: {len(batches)} × {LABEL_BATCH_SIZE} topics\n"
        f"  Arbiter decisions: {len(council_results)}/{len(top_topics)}\n"
        f"  Model wins → {model_wins}  |  Synthesised: {synth_count}\n"
        f"  Sample:\n{sample_text}\n"
        f"  Checkpoint: labels_{run_key}.json\n"
        f"Review table ready. Set Approve/Rename, then click Submit Review."
    )


@tool
def run_bertopic_and_label(
    run_key: str, threshold: float = AGGLO_DISTANCE_THRESHOLD
) -> str:
    """Run AgglomerativeClustering-based BERTopic discovery AND council-of-3-LLMs labelling
    in one step. run_key must be 'abstract', 'title', or 'combined'.
    threshold is the Euclidean distance_threshold in 10d UMAP space (default 1.5).
    Lower values → more, finer clusters. Higher values → fewer, coarser clusters."""
    # FIX (2026-06-05): outer try/except so ANY exception returns a string rather
    # than escaping the @tool boundary.  An unhandled exception here writes a
    # dangling AIMessage(tool_calls=[...]) to MemorySaver with no matching
    # ToolMessage, permanently breaking the LangGraph thread.
    try:
        # ── Full-phase cache ──────────────────────────────────────────────────
        # If labels already exist (from a previous run committed to git, or
        # restored from a bundle), skip the entire pipeline and return immediately.
        # This is the key mechanism that makes redeploys safe.
        labels_path = CHECKPOINT_DIR / f"labels_{run_key}.json"
        if labels_path.exists():
            try:
                existing = json.loads(labels_path.read_text())
                if existing:
                    n = len(existing)
                    print(f"[BERTopic] Cache hit — labels_{run_key}.json has {n} entries, skipping re-run.")
                    return (
                        f"✅ Labels already computed for run_key='{run_key}' ({n} topics).\n"
                        f"Loaded from cache — no re-run needed.\n"
                        f"Ready for Phase 3: call consolidate_into_themes(run_key='{run_key}', theme_map={{...}})"
                    )
            except Exception:
                pass  # corrupted file — fall through to re-run

        discovery_result = run_bertopic_discovery(run_key, threshold)
        label_result = label_topics_with_llm(run_key)
        return discovery_result + "\n\n" + label_result
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[run_bertopic_and_label] ERROR:\n{tb}")
        return (
            f"❌ run_bertopic_and_label failed on run_key='{run_key}'.\n"
            f"Error: {type(exc).__name__}: {exc}\n\n"
            f"Partial traceback (last 800 chars):\n{tb[-800:]}\n\n"
            f"You may retry with the same arguments, or try a different run_key."
        )


# ─── Tool 3: Consolidate into Themes ─────────────────────────────────────────


@tool
def consolidate_into_themes(run_key: str, theme_map: dict) -> str:
    """Merge researcher-approved topic groups into consolidated themes.
    theme_map is a dict: {"Theme Name": [cluster_id1, cluster_id2, ...], ...}
    Recomputes sentence/paper counts. Saves themes.json."""
    try:
        assert (CHECKPOINT_DIR / f"labels_{run_key}.json").exists(), (
            f"labels_{run_key}.json not found. Call run_bertopic_and_label(run_key='{run_key}') first."
        )
        assert (CHECKPOINT_DIR / f"emb_{run_key}.npy").exists(), (
            f"emb_{run_key}.npy not found. Call run_bertopic_and_label(run_key='{run_key}') first."
        )
        assert (CHECKPOINT_DIR / f"sentences_{run_key}.json").exists(), (
            f"sentences_{run_key}.json not found. Call run_bertopic_and_label(run_key='{run_key}') first."
        )

        labels_data = json.loads((CHECKPOINT_DIR / f"labels_{run_key}.json").read_text())
        cluster_lookup = dict(map(lambda t: (str(t["cluster_id"]), t), labels_data))

        def build_theme(theme_name_ids):
            theme_name, cluster_ids = theme_name_ids
            member_topics = list(
                filter(None, map(lambda cid: cluster_lookup.get(str(cid)), cluster_ids))
            )
            all_sentences = list(
                ichain.from_iterable(
                    map(lambda t: t.get("top_sentences", []), member_topics)
                )
            )
            total_size = sum(map(lambda t: t.get("size", 0), member_topics))
            topic_labels = list(
                map(lambda t: t.get("label", f"T{t['cluster_id']}"), member_topics)
            )
            all_paper_indices = set(
                ichain.from_iterable(
                    map(lambda t: t.get("paper_indices", []), member_topics)
                )
            )
            return {
                "theme_name": theme_name,
                "cluster_ids": cluster_ids,
                "merged_topic_labels": topic_labels,
                "total_sentences": total_size,
                "paper_count": len(all_paper_indices),
                "representative_sentences": all_sentences[:NEAREST_K],
                "sub_topics": len(member_topics),
            }

        themes = list(map(build_theme, theme_map.items()))
        themes_sorted = sorted(themes, key=lambda x: x["total_sentences"], reverse=True)

        (CHECKPOINT_DIR / f"themes_{run_key}.json").write_text(
            json.dumps(themes_sorted, indent=2)
        )
        theme_summary = "\n".join(
            list(
                map(
                    lambda pair: (
                        f"  {pair[0] + 1}. {pair[1]['theme_name']} ({pair[1]['total_sentences']} sentences, {pair[1]['sub_topics']} sub-topics)"
                    ),
                    enumerate(themes_sorted),
                )
            )
        )

        return (
            f"Themes consolidated for run_key='{run_key}'.\n"
            f"  Total themes: {len(themes_sorted)}\n"
            f"Themes:\n{theme_summary}\n"
            f"  Checkpoint: themes_{run_key}.json\n"
            f"Review consolidated themes in the table. Click Submit Review to confirm or adjust."
        )
    except Exception as exc:
        return f"❌ consolidate_into_themes failed: {type(exc).__name__}: {exc}"


# ─── Tool 4: Compare with PAJAIS Taxonomy ────────────────────────────────────


@tool
def compare_with_taxonomy(run_key: str) -> str:
    """Map final themes to the PAJAIS 25 research categories using ChatGroq LLM.
    Each theme is classified as MAPPED (with category) or NOVEL (new contribution).
    Saves taxonomy_map.json."""
    try:
        assert (CHECKPOINT_DIR / f"themes_{run_key}.json").exists(), (
            f"themes_{run_key}.json not found. Call consolidate_into_themes(run_key='{run_key}') first."
        )

        themes = json.loads((CHECKPOINT_DIR / f"themes_{run_key}.json").read_text())

        from agent import KEY_ROTATOR

        llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=SecretStr(KEY_ROTATOR.next()),
            temperature=0.1,
            max_retries=5,
        )

        categories_text = "\n".join(
            list(
                map(
                    lambda pair: f"  {pair[0] + 1}. {pair[1]}", enumerate(PAJAIS_CATEGORIES)
                )
            )
        )

        def build_theme_text(t):
            return f"THEME: {t['theme_name']}\nSentences: " + "; ".join(
                t.get("representative_sentences", [])[:2]
            )

        prompt = PromptTemplate.from_template(
            "You are a systematic literature review expert. Map each research theme to the PAJAIS taxonomy.\n\n"
            "PAJAIS Categories:\n{categories}\n\n"
            "For each theme, return a JSON object with:\n"
            '- "theme_name": string (exact match from input)\n'
            '- "pajais_match": string (exact category name from list above, or "NOVEL" if no match)\n'
            '- "match_confidence": float 0-1 (0 if NOVEL)\n'
            '- "is_novel": boolean\n'
            '- "reasoning": one sentence explanation of the mapping or why it is novel\n\n'
            "Return ONLY a JSON array, no preamble, no markdown, no backticks.\n\n"
            "Themes to map:\n{themes_text}"
        )

        chain = prompt | llm | _make_json_parser()  # FIX: robust parser

        BATCH_SIZE = 20

        def make_batches(lst, size):
            return list(map(lambda i: lst[i : i + size], range(0, len(lst), size)))

        def process_batch(batch):
            try:
                themes_text = "\n\n".join(list(map(build_theme_text, batch)))
                res = chain.invoke({"categories": categories_text, "themes_text": themes_text})
                time.sleep(2)
                return res
            except Exception as exc:
                print(f"[TaxonomyBatch] failed: {type(exc).__name__}: {exc}")
                return []

        batches = make_batches(themes, BATCH_SIZE)
        raw_results = list(map(process_batch, batches))

        result = list(
            filter(
                lambda item: isinstance(item, dict) and "theme_name" in item,
                ichain.from_iterable(filter(lambda x: isinstance(x, list), raw_results)),
            )
        )
        mapped = dict(map(lambda item: (item["theme_name"], item), result))

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
                            "reasoning": "Taxonomy validation threshold unmet",
                        },
                    ),
                },
                themes,
            )
        )

        novel_count = sum(map(lambda t: int(t.get("is_novel", False)), enriched))
        mapped_count = len(enriched) - novel_count

        (CHECKPOINT_DIR / f"taxonomy_map_{run_key}.json").write_text(
            json.dumps(enriched, indent=2)
        )

        novel_names = list(
            map(
                lambda t: t["theme_name"],
                filter(lambda t: t.get("is_novel", False), enriched),
            )
        )
        novel_text = (
            "\n".join(map(lambda n: f"  * {n}", novel_names)) if novel_names else "  (none)"
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
    except Exception as exc:
        return f"❌ compare_with_taxonomy failed: {type(exc).__name__}: {exc}"


# ─── Tool 5: Generate Comparison CSV ─────────────────────────────────────────


@tool
def generate_comparison_csv() -> str:
    """Load themes from both abstract and title runs and build a semantically-matched
    side-by-side comparison DataFrame. Uses cosine similarity on theme name embeddings
    to pair each abstract theme with its best-matching title theme.
    Saves comparison.csv."""
    try:
        abstract_path = CHECKPOINT_DIR / "themes_abstract.json"
        title_path = CHECKPOINT_DIR / "themes_title.json"

        abstract_themes = (
            json.loads(abstract_path.read_text()) if abstract_path.exists() else []
        )
        title_themes = json.loads(title_path.read_text()) if title_path.exists() else []

        CONVERGENCE_THRESHOLD = 0.40
        model = SentenceTransformer(EMBED_MODEL)

        abstract_names = list(map(lambda t: t.get("theme_name", ""), abstract_themes))
        title_names = list(map(lambda t: t.get("theme_name", ""), title_themes))

        abstract_embs = (
            model.encode(abstract_names, normalize_embeddings=True)
            if abstract_names
            else np.zeros((0, 768))
        )
        title_embs = (
            model.encode(title_names, normalize_embeddings=True)
            if title_names
            else np.zeros((0, 768))
        )

        used_title_idxs = set()

        def match_abstract_theme(abs_idx):
            return (None, 0.0) if len(title_embs) == 0 else match_abstract_theme_v(abs_idx)

        def match_abstract_theme_v(abs_idx):
            sims = cosine_similarity(abstract_embs[abs_idx : abs_idx + 1], title_embs)[0]
            available_mask = np.array(
                list(map(lambda i: 0.0 if i in used_title_idxs else 1.0, range(len(sims))))
            )
            masked_sims = sims * available_mask
            best_idx = int(masked_sims.argmax())
            return best_idx, float(masked_sims[best_idx])

        matched_pairs = list(
            map(
                lambda abs_idx: (abs_idx, *match_abstract_theme(abs_idx)),
                range(len(abstract_themes)),
            )
        )

        def claim_match(triple):
            return (
                (used_title_idxs.add(triple[1]) or (triple[0], triple[1], triple[2]))
                if (triple[1] is not None and triple[1] not in used_title_idxs)
                else (triple[0], None, triple[2])
            )

        final_pairs = list(map(claim_match, matched_pairs))

        def make_row(triple):
            abs_idx, title_idx, sim = triple
            at = abstract_themes[abs_idx]
            tt = title_themes[title_idx] if title_idx is not None else {}
            return {
                "Abstract Theme": at.get("theme_name", ""),
                "Abstract Sentences": at.get("total_sentences", ""),
                "Abstract Papers": at.get("paper_count", ""),
                "Title Theme": tt.get("theme_name", ""),
                "Title Sentences": tt.get("total_sentences", ""),
                "Title Papers": tt.get("paper_count", ""),
                "Similarity": round(sim, 3) if tt.get("theme_name", "") else "",
                "Convergence": "ABSTRACT ONLY"
                if not tt.get("theme_name", "")
                else ("CONVERGED" if sim >= CONVERGENCE_THRESHOLD else "DIVERGED"),
            }

        rows = list(map(make_row, final_pairs))
        matched_title_idxs = set(filter(None, map(lambda triple: triple[1], final_pairs)))
        unmatched_title_rows = list(
            map(
                lambda ti: {
                    "Abstract Theme": "",
                    "Abstract Sentences": "",
                    "Abstract Papers": "",
                    "Title Theme": title_themes[ti].get("theme_name", ""),
                    "Title Sentences": title_themes[ti].get("total_sentences", ""),
                    "Title Papers": title_themes[ti].get("paper_count", ""),
                    "Similarity": "",
                    "Convergence": "TITLE ONLY",
                },
                filter(lambda i: i not in matched_title_idxs, range(len(title_themes))),
            )
        )

        df = pd.DataFrame(rows + unmatched_title_rows)
        df.to_csv(CHECKPOINT_DIR / "comparison.csv", index=False)

        converged = sum(
            map(lambda r: int(r["Convergence"] == "CONVERGED"), df.to_dict("records"))
        )
        abstract_only = sum(
            map(lambda r: int(r["Convergence"] == "ABSTRACT ONLY"), df.to_dict("records"))
        )
        title_only = sum(
            map(lambda r: int(r["Convergence"] == "TITLE ONLY"), df.to_dict("records"))
        )
        diverged = sum(
            map(lambda r: int(r["Convergence"] == "DIVERGED"), df.to_dict("records"))
        )

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
    except Exception as exc:
        return f"❌ generate_comparison_csv failed: {type(exc).__name__}: {exc}"


# ─── Tool 6: Export Narrative ─────────────────────────────────────────────────


@tool
def export_narrative(run_key: str = "abstract") -> str:
    """Generate a 500-word Section 7 narrative draft for a literature review paper.
    Uses final themes and PAJAIS taxonomy mapping. Saves narrative.txt."""
    try:
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

        tax_lookup = dict(map(lambda t: (t["theme_name"], t), taxonomy))

        def theme_summary(t):
            return f"- {t['theme_name']} ({t['total_sentences']} sentences) " + (
                "[NOVEL]"
                if tax_lookup.get(t["theme_name"], {}).get("is_novel", False)
                else f"[PAJAIS: {tax_lookup.get(t['theme_name'], {}).get('pajais_match', 'NOVEL')}]"
            )

        themes_summary = "\n".join(list(map(theme_summary, themes)))
        novel_themes = list(
            map(
                lambda t: t["theme_name"],
                filter(
                    lambda t: tax_lookup.get(t["theme_name"], {}).get("is_novel", False),
                    themes,
                ),
            )
        )
        novel_list = ", ".join(novel_themes) if novel_themes else "none identified"

        from agent import KEY_ROTATOR

        llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=SecretStr(KEY_ROTATOR.next()),
            temperature=0.3,
            max_retries=5,
        )
        prompt = PromptTemplate.from_template(
            "You are writing Section 7 of a conference paper on topic modelling of a journal's literature.\n\n"
            "Write approximately 500 words. Structure as follows:\n"
            "(a) Methodology: State that BERTopic with sentence-level embeddings (allenai/specter2_base, 768 dimensions), "
            "UMAP dimensionality reduction (10d), and AgglomerativeClustering (Ward linkage, Euclidean distance) was used. "
            "Mention Braun & Clarke (2006) six-phase thematic analysis and the researcher-in-the-loop validation.\n"
            "(b) Findings: Describe each theme with its label, sentence count, and PAJAIS mapping status. "
            "Reference the comparison CSV and PAJAIS taxonomy map.\n"
            "(c) Interpretation: What do these themes reveal about the journal's research landscape? "
            "Highlight convergence between abstract and title runs as evidence of stability.\n"
            "(d) Contribution: Explicitly name the NOVEL themes as the paper's contribution to the field.\n"
            "(e) Limitations: Acknowledge UMAP stochasticity, sentence-level (not document-level) analysis, "
            "and LLM labelling subjectivity per Carlsen & Ralund (2022).\n\n"
            "Themes discovered (run: {run_key}):\n{themes_summary}\n\n"
            "Novel themes (not in PAJAIS taxonomy): {novel_list}\n\n"
            "Write in formal academic English. Cite Braun & Clarke (2006), Grootendorst (2022), and Carlsen & Ralund (2022)."
        )

        chain = prompt | llm
        response = chain.invoke(
            {"run_key": run_key, "themes_summary": themes_summary, "novel_list": novel_list}
        )

        narrative = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
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
    except Exception as exc:
        return f"❌ export_narrative failed: {type(exc).__name__}: {exc}"


# ─── Tool 7: Checkpoint diagnostics ──────────────────────────────────────────


@tool
def check_checkpoints() -> str:
    """Inspect the checkpoints directory and report exactly what has been saved.
    Use this whenever the review table appears empty or the researcher asks what
    state the pipeline is in. Returns file names, sizes, and entry counts."""
    try:
        if not CHECKPOINT_DIR.exists():
            return "❌ Checkpoint directory does not exist yet. Has Phase 1 run?"

        all_files = sorted(CHECKPOINT_DIR.iterdir())
        if not all_files:
            return "Checkpoint directory exists but is empty. Has Phase 1 run?"

        lines = ["Checkpoint state:"]
        for f in all_files:
            size_kb = f.stat().st_size / 1024
            if f.suffix == ".json":
                try:
                    data = json.loads(f.read_text())
                    entry_count = len(data) if isinstance(data, (list, dict)) else "?"
                    lines.append(f"  {f.name}  ({size_kb:.1f} KB, {entry_count} entries)")
                except Exception:
                    lines.append(f"  {f.name}  ({size_kb:.1f} KB, unreadable)")
            else:
                lines.append(f"  {f.name}  ({size_kb:.1f} KB)")

        # Summarise what's ready for the review table
        lines.append("")
        lines.append("Review table availability:")
        for run_key in ("abstract", "title", "combined"):
            for phase, fname in [
                ("taxonomy", f"taxonomy_map_{run_key}.json"),
                ("themes", f"themes_{run_key}.json"),
                ("labels", f"labels_{run_key}.json"),
            ]:
                p = CHECKPOINT_DIR / fname
                if p.exists():
                    try:
                        n = len(json.loads(p.read_text()))
                    except Exception:
                        n = 0
                    lines.append(
                        f"  run_key='{run_key}' → {phase} ready ({n} entries) — "
                        f"set the dropdown to '{run_key}' and click Refresh table"
                        if n > 0
                        else f"  run_key='{run_key}' → {phase} file exists but is EMPTY"
                    )
                    break  # only report the most advanced phase

        return "\n".join(lines)
    except Exception as exc:
        return f"❌ check_checkpoints failed: {type(exc).__name__}: {exc}"


# ─── Exported tool list ───────────────────────────────────────────────────────

ALL_TOOLS = [
    load_scopus_csv,
    run_bertopic_and_label,
    consolidate_into_themes,
    compare_with_taxonomy,
    generate_comparison_csv,
    export_narrative,
    check_checkpoints,
]