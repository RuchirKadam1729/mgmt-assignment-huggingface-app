#!/usr/bin/env python3
"""
package_submission.py
Packages existing pipeline checkpoint outputs into the RQ5/RQ6/RQ7 submission structure.
Run from project root AFTER the full pipeline has completed.

Usage:
    python package_submission.py
"""

import csv
import json
import shutil
from pathlib import Path

from reportlab.lib import colors

# pip install reportlab --break-system-packages (or uv add reportlab)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

CHECKPOINTS = Path("checkpoints")
OUT = Path("submission")
W, H = A4

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=12)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=8)
BODY = ParagraphStyle(
    "Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6
)
CODE = ParagraphStyle(
    "Code",
    parent=styles["Code"],
    fontSize=8,
    leading=11,
    backColor=colors.HexColor("#F5F5F5"),
)


def mkdir(p):
    p.mkdir(parents=True, exist_ok=True)
    return p


def load(name):
    p = CHECKPOINTS / name
    return json.loads(p.read_text()) if p.exists() else None


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ {path.name}")


def copy(src_name, dst):
    src = CHECKPOINTS / src_name
    if src.exists():
        shutil.copy(src, dst)
        print(f"  ✓ {dst.name}")
    else:
        print(f"  ! MISSING {src_name} (skipped)")


def pdf(path, story):
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )
    doc.build(story)
    print(f"  ✓ {path.name}")


# ─── Shared code stubs ───────────────────────────────────────────────────────

PREPROCESSING_PY = '''\
#!/usr/bin/env python3
"""RQ5 — Preprocessing: boilerplate removal and sentence splitting."""
import re, pandas as pd
from nltk.tokenize import sent_tokenize

BOILERPLATE_PATTERNS = [
    r"©\\s*\\d{4}.*", r"All rights reserved.*", r"Published by Elsevier.*",
    r"doi:.*", r"https?://\\S+", r"^\\s*\\d+\\s*$",
]

def clean(text):
    return re.sub("|".join(BOILERPLATE_PATTERNS), " ", text, flags=re.IGNORECASE).strip()

def split_sentences(texts):
    return [s.strip() for t in texts for s in sent_tokenize(clean(t)) if len(s.strip()) > 30]

def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    abstracts = list(df["Abstract"].dropna().astype(str)) if "Abstract" in df.columns else []
    return split_sentences(abstracts)
'''

EMBEDDING_PY = '''\
#!/usr/bin/env python3
"""RQ5 — Embedding: allenai/specter2_base (768-dimensional scientific embeddings)."""
from sentence_transformers import SentenceTransformer
import numpy as np

EMBED_MODEL = "allenai/specter2_base"  # 768d, trained on scientific papers

def embed(sentences: list[str]) -> np.ndarray:
    model = SentenceTransformer(EMBED_MODEL)
    return model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)
'''

CLUSTERING_PY = '''\
#!/usr/bin/env python3
"""RQ5 — Clustering: UMAP (768d→10d) + AgglomerativeClustering (Ward/Euclidean)."""
import numpy as np
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

DISTANCE_THRESHOLD = 1.5   # Euclidean distance in 10d UMAP space
MIN_CLUSTER_SIZE   = 20    # discard clusters below this (noise)

def reduce_and_cluster(embeddings: np.ndarray):
    reducer = UMAP(n_components=10, n_neighbors=15,
                   min_dist=0.0, metric="cosine", random_state=42)
    reduced = reducer.fit_transform(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=None, metric="euclidean",
        linkage="ward", distance_threshold=DISTANCE_THRESHOLD
    )
    raw_labels = clustering.fit_predict(reduced)

    sizes = Counter(raw_labels.tolist())
    valid = {cid for cid, sz in sizes.items() if sz >= MIN_CLUSTER_SIZE}
    return np.array([c if c in valid else -1 for c in raw_labels])
'''

AI_COUNCIL_PY = '''\
#!/usr/bin/env python3
"""RQ6 — AI Council: 3 independent LLM labellers + arbiter."""
# Council models (Groq free tier)
COUNCIL_MODELS = [
    "llama-3.3-70b-versatile",   # primary  — high quality
    "llama-3.1-8b-instant",       # fast lightweight voice
    "gemma2-9b-it",               # third independent vote
]
ARBITER_MODEL    = "llama-3.3-70b-versatile"
LABEL_BATCH_SIZE = 15   # topics per API call (keeps under 6k TPM free-tier limit)

# Each model independently labels every topic cluster given its top-3 evidence sentences.
# The arbiter picks / synthesises the best label based on precision, academic tone,
# and distinctiveness. winning_model field: 1=Llama-70b, 2=Llama-8b, 3=Gemma, 0=synthesised.
# Full implementation: label_topics_with_llm() in tools.py
'''

LABELING_PY = '''\
#!/usr/bin/env python3
"""RQ6 — Labeling pipeline orchestrator."""
# Stage 1: run_bertopic_discovery()  → summaries_{run}.json (cluster centroids, top sentences)
# Stage 2: label_topics_with_llm()   → labels_{run}.json    (AI council labels + proposals)
# Stage 3: consolidate_into_themes() → themes_{run}.json    (researcher-approved groupings)
# Stage 4: compare_with_taxonomy()   → taxonomy_map_{run}.json (PAJAIS mapping)
# Full pipeline exposed as @tool via LangGraph ReAct agent in agent.py / tools.py
'''

TCCM_PY = '''\
#!/usr/bin/env python3
"""RQ6 — PAJAIS taxonomy classifier (Jiang et al., 2019)."""
PAJAIS_CATEGORIES = [
    "IS Strategy and Management",       "IT Adoption and Diffusion",
    "E-Commerce and Digital Markets",   "Business Intelligence and Analytics",
    "Knowledge Management",             "IT and Organizational Change",
    "Supply Chain and ERP Systems",     "Healthcare IT",
    "Mobile and Wireless Computing",    "Social Media and Web 2.0",
    "Security and Privacy",             "Cloud Computing",
    "Big Data and Data Mining",         "Artificial Intelligence and Machine Learning",
    "Human-Computer Interaction",       "Software Development and Engineering",
    "IT Governance and Outsourcing",    "Digital Innovation and Entrepreneurship",
    "IS Education and Research Methods","Collaborative Systems and CSCW",
    "Open Source and Crowdsourcing",    "IT and Society",
    "Financial Technology",             "Smart Cities and IoT",
    "Agile and DevOps Practices",
]
# Each theme is mapped via ChatGroq LLM prompt.
# NOVEL = theme has no close PAJAIS match → represents a research gap / contribution.
# Full implementation: compare_with_taxonomy() in tools.py
'''

# ─── RQ5 ─────────────────────────────────────────────────────────────────────


def build_rq5():
    print("\n── RQ5: Latent Topic Discovery ──")
    rq5 = mkdir(OUT / "RQ5")

    (rq5 / "preprocessing.py").write_text(PREPROCESSING_PY)
    print("  ✓ preprocessing.py")
    (rq5 / "embedding.py").write_text(EMBEDDING_PY)
    print("  ✓ embedding.py")
    (rq5 / "clustering.py").write_text(CLUSTERING_PY)
    print("  ✓ clustering.py")

    # cluster_summary.csv
    labels = load("labels_abstract.json")
    if labels:
        write_csv(
            rq5 / "cluster_summary.csv",
            [
                {
                    "cluster_id": t["cluster_id"],
                    "label": t.get("label", ""),
                    "sentence_count": t.get("size", ""),
                    "paper_count": t.get("paper_count", ""),
                    "confidence": round(t.get("confidence", 0), 3),
                    "winning_model": t.get("winning_model", ""),
                    "top_evidence": (t.get("top_sentences", [""])[0])[:120],
                }
                for t in labels
            ],
            [
                "cluster_id",
                "label",
                "sentence_count",
                "paper_count",
                "confidence",
                "winning_model",
                "top_evidence",
            ],
        )

        # paper_results.csv
        data_df = None
        if (CHECKPOINTS / "data.csv").exists():
            import pandas as pd

            data_df = pd.read_csv(CHECKPOINTS / "data.csv")
        write_csv(
            rq5 / "paper_results.csv",
            [
                {
                    "paper_index": pid,
                    "cluster_id": t["cluster_id"],
                    "cluster_label": t.get("label", ""),
                    "title": (
                        data_df.iloc[pid]["Title"]
                        if data_df is not None and pid < len(data_df)
                        else ""
                    ),
                    "year": (
                        data_df.iloc[pid]["Year"]
                        if data_df is not None and pid < len(data_df)
                        else ""
                    ),
                }
                for t in labels
                for pid in t.get("paper_indices", [])
            ],
            ["paper_index", "cluster_id", "cluster_label", "title", "year"],
        )

    copy("chart_map_abstract.html", rq5 / "umap_plot.html")
    copy("chart_bar_abstract.html", rq5 / "topic_sizes_bar.html")

    # methodology.pdf
    story = [
        Paragraph("RQ4 / RQ5: Latent Topic Discovery — Methodology", H1),
        Paragraph(
            "Braun &amp; Clarke (2006) Six-Phase Thematic Analysis | BERTopic Pipeline",
            H2,
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("<b>Embedding Model</b>", H2),
        Paragraph(
            "allenai/specter2_base — 768-dimensional sentence embeddings pre-trained on scientific paper abstracts and citations (Specter2, Lo et al., 2020). Each sentence is encoded to a unit-normalised 768d vector capturing domain-specific semantic relationships.",
            BODY,
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("<b>Dimensionality Reduction</b>", H2),
        Paragraph(
            "UMAP (Uniform Manifold Approximation and Projection) reduces 768d embeddings to 10 dimensions (n_neighbors=15, min_dist=0.0, metric=cosine, random_state=42). This separates semantic clusters that would otherwise be obscured by the curse of dimensionality in high-dimensional cosine space.",
            BODY,
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("<b>Clustering</b>", H2),
        Paragraph(
            "AgglomerativeClustering with Ward linkage and Euclidean distance in 10d UMAP space (distance_threshold=1.5). Unlike DBSCAN, Ward linkage minimises within-cluster variance and avoids the chaining problem that collapses all academic sentences into a single cluster. Clusters smaller than 20 sentences are discarded as noise.",
            BODY,
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("<b>Topic Labelling — Council of 3 LLMs</b>", H2),
        Paragraph(
            "Three language models (Llama-3.3-70b-versatile, Llama-3.1-8b-instant, Gemma2-9b-it) independently label each cluster given its top-3 centroid-nearest evidence sentences. A fourth arbiter model (Llama-3.3-70b) selects or synthesises the best label based on precision, academic terminology, and distinctiveness. Calls are batched at 15 topics per request to remain within Groq free-tier rate limits (6,000 TPM).",
            BODY,
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("<b>Researcher Validation</b>", H2),
        Paragraph(
            "Following Braun &amp; Clarke (2006), the researcher reviews all machine-generated labels via an interactive table, approving, renaming, or merging topics before proceeding. This human-in-the-loop gate is enforced at four checkpoints in the pipeline.",
            BODY,
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("<b>References</b>", H2),
        Paragraph(
            "Braun, V. &amp; Clarke, V. (2006). Using thematic analysis in psychology. Qualitative Research in Psychology, 3(2), 77–101.",
            BODY,
        ),
        Paragraph(
            "Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.",
            BODY,
        ),
        Paragraph(
            "Carlsen, H.B. &amp; Ralund, S. (2022). Computational grounded theory revisited. Big Data &amp; Society.",
            BODY,
        ),
    ]
    pdf(rq5 / "methodology.pdf", story)


# ─── RQ6 ─────────────────────────────────────────────────────────────────────


def build_rq6():
    print("\n── RQ6: AI-Augmented Paper Classification ──")
    rq6 = mkdir(OUT / "RQ6")

    (rq6 / "ai_council.py").write_text(AI_COUNCIL_PY)
    print("  ✓ ai_council.py")
    (rq6 / "labeling.py").write_text(LABELING_PY)
    print("  ✓ labeling.py")
    (rq6 / "tccm_classifier.py").write_text(TCCM_PY)
    print("  ✓ tccm_classifier.py")

    tax = load("taxonomy_map_abstract.json")
    if tax:
        write_csv(
            rq6 / "cluster_summary.csv",
            [
                {
                    "theme_name": t.get("theme_name", ""),
                    "pajais_category": t.get("pajais_match", "NOVEL"),
                    "is_novel": t.get("is_novel", False),
                    "confidence": round(t.get("match_confidence", 0), 3),
                    "sentence_count": t.get("total_sentences", ""),
                    "paper_count": t.get("paper_count", ""),
                    "sub_topics": t.get("sub_topics", ""),
                    "reasoning": t.get("reasoning", ""),
                }
                for t in tax
            ],
            [
                "theme_name",
                "pajais_category",
                "is_novel",
                "confidence",
                "sentence_count",
                "paper_count",
                "sub_topics",
                "reasoning",
            ],
        )

    copy("chart_heat_abstract.html", rq6 / "similarity_heatmap.html")

    # classification_results.pdf
    story = [
        Paragraph("RQ6: AI-Augmented Paper Classification — Results", H1),
        Paragraph("PAJAIS Taxonomy Mapping (Jiang et al., 2019)", H2),
        Spacer(1, 0.3 * cm),
    ]
    if tax:
        mapped = [t for t in tax if not t.get("is_novel", False)]
        novel = [t for t in tax if t.get("is_novel", False)]
        story += [
            Paragraph(
                f"Total themes: <b>{len(tax)}</b> &nbsp;|&nbsp; "
                f"Mapped to PAJAIS: <b>{len(mapped)}</b> &nbsp;|&nbsp; "
                f"Novel (gap): <b>{len(novel)}</b>",
                BODY,
            ),
            Spacer(1, 0.4 * cm),
            Paragraph("<b>Mapped Themes</b>", H2),
        ]
        for t in mapped:
            story.append(
                Paragraph(
                    f"<b>{t.get('theme_name', '')}</b> → {t.get('pajais_match', '')} "
                    f"(conf={round(t.get('match_confidence', 0), 2)}, "
                    f"{t.get('total_sentences', '')} sentences, {t.get('paper_count', '')} papers)<br/>"
                    f"<i>{t.get('reasoning', '')}</i>",
                    BODY,
                )
            )

        story += [
            Spacer(1, 0.4 * cm),
            Paragraph("<b>Novel Themes — Research Gaps</b>", H2),
        ]
        for t in novel:
            story.append(
                Paragraph(
                    f"<b>{t.get('theme_name', '')}</b> — NOT in PAJAIS taxonomy "
                    f"({t.get('total_sentences', '')} sentences, {t.get('paper_count', '')} papers)<br/>"
                    f"<i>{t.get('reasoning', '')}</i>",
                    BODY,
                )
            )
    else:
        story.append(
            Paragraph(
                "taxonomy_map_abstract.json not found. Run Phase 5.5 first.", BODY
            )
        )

    story += [
        Spacer(1, 0.4 * cm),
        Paragraph("<b>Classification Method</b>", H2),
        Paragraph(
            "Each theme (produced by BERTopic + researcher validation) is classified against the 25-category PAJAIS taxonomy using an LLM prompt (Llama-3.3-70b-versatile, temperature=0.1). Themes with no suitable PAJAIS match are flagged NOVEL and constitute the paper's original contribution.",
            BODY,
        ),
        Paragraph(
            "Reference: Jiang, J.J. et al. (2019). A classification scheme for IS research. PAJAIS.",
            BODY,
        ),
    ]
    pdf(rq6 / "classification_results.pdf", story)


# ─── RQ7 ─────────────────────────────────────────────────────────────────────


def build_rq7():
    print("\n── RQ7: Future Research Agenda ──")
    rq7 = mkdir(OUT / "RQ7")
    tccm = mkdir(rq7 / "tccm_outputs")

    copy("taxonomy_map_abstract.json", tccm / "taxonomy_map_abstract.json")
    copy("themes_abstract.json", tccm / "themes_abstract.json")
    copy("labels_abstract.json", tccm / "labels_abstract.json")

    labels = load("labels_abstract.json")
    if labels:
        import pandas as pd

        data_df = (
            pd.read_csv(CHECKPOINTS / "data.csv")
            if (CHECKPOINTS / "data.csv").exists()
            else None
        )
        write_csv(
            rq7 / "paper_results.csv",
            [
                {
                    "paper_index": pid,
                    "cluster_id": t["cluster_id"],
                    "cluster_label": t.get("label", ""),
                    "title": (
                        data_df.iloc[pid]["Title"]
                        if data_df is not None and pid < len(data_df)
                        else ""
                    ),
                    "year": (
                        data_df.iloc[pid]["Year"]
                        if data_df is not None and pid < len(data_df)
                        else ""
                    ),
                }
                for t in labels
                for pid in t.get("paper_indices", [])
            ],
            ["paper_index", "cluster_id", "cluster_label", "title", "year"],
        )

    tax = load("taxonomy_map_abstract.json")
    if tax:
        write_csv(
            rq7 / "cluster_summary.csv",
            [
                {
                    "theme_name": t.get("theme_name", ""),
                    "pajais_category": t.get("pajais_match", "NOVEL"),
                    "is_novel": t.get("is_novel", False),
                    "sentence_count": t.get("total_sentences", ""),
                    "paper_count": t.get("paper_count", ""),
                }
                for t in tax
            ],
            [
                "theme_name",
                "pajais_category",
                "is_novel",
                "sentence_count",
                "paper_count",
            ],
        )

    # agenda_synthesis.pdf
    narrative = (
        (CHECKPOINTS / "narrative.txt").read_text()
        if (CHECKPOINTS / "narrative.txt").exists()
        else ""
    )
    novel_themes = [
        t.get("theme_name", "") for t in (tax or []) if t.get("is_novel", False)
    ]

    story = [
        Paragraph("RQ7: Future Research Agenda Synthesis", H1),
        Paragraph("Prescriptive Agenda Setting via PAJAIS Gap Analysis", H2),
        Spacer(1, 0.3 * cm),
        Paragraph("<b>Research Gaps Identified (Novel Themes)</b>", H2),
        Paragraph(
            "The following themes emerged from the corpus but are absent from the PAJAIS taxonomy, indicating underexplored areas that constitute a future research agenda:",
            BODY,
        ),
        Spacer(1, 0.2 * cm),
    ]
    for i, theme in enumerate(novel_themes, 1):
        story.append(Paragraph(f"{i}. <b>{theme}</b>", BODY))

    if narrative:
        story += [
            Spacer(1, 0.5 * cm),
            Paragraph("<b>Section 7 — Narrative Synthesis</b>", H2),
            Spacer(1, 0.2 * cm),
        ]
        for para in narrative.split("\n\n"):
            clean = (
                para.strip()
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            if clean:
                story.append(Paragraph(clean, BODY))
    else:
        story.append(
            Paragraph(
                "narrative.txt not found — run Phase 6 (export_narrative) first.", BODY
            )
        )

    story += [
        Spacer(1, 0.4 * cm),
        Paragraph("<b>Methodology Note</b>", H2),
        Paragraph(
            "Novel themes were identified by mapping all discovered BERTopic themes against the 25-category PAJAIS taxonomy. Themes with no PAJAIS match (is_novel=True) represent research areas not yet codified in the IS literature classification, forming the basis of this prescriptive agenda.",
            BODY,
        ),
    ]
    pdf(rq7 / "agenda_synthesis.pdf", story)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Submission Packager — BERTopic Agentic AI")
    print("=" * 55)
    build_rq5()
    build_rq6()
    build_rq7()

    print("\n" + "=" * 55)
    print("  Done. Submission folder structure:")
    print("=" * 55)
    for p in sorted(OUT.rglob("*")):
        depth = len(p.relative_to(OUT).parts)
        print("  " + "  " * (depth - 1) + p.name)
