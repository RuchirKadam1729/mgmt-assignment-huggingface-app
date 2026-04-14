---
title: BERTopic Agentic AI
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.6.0
app_file: app.py
pinned: false
license: mit
---

# BERTopic Agentic AI

**Braun & Clarke (2006) · 6-phase thematic analysis · PAJAIS taxonomy mapping**

An agentic AI application that performs end-to-end computational thematic analysis on Scopus literature export CSVs.

## Setup

1. Set your `GROQ_API_KEY` in **Settings → Secrets** of this Space.
2. Upload your Scopus CSV (file named like `[JournalName]_TopicModelling_Export.csv`).
3. Type `run abstract` in the chat to begin.

## Pipeline

| Phase | Name | Tool |
|-------|------|------|
| 1 | Familiarisation | `load_scopus_csv` |
| 2 | Initial Codes | `run_bertopic_and_label` |
| 3 | Searching Themes | `consolidate_into_themes` |
| 4 | Reviewing Themes | Saturation check |
| 5.5 | PAJAIS Mapping | `compare_with_taxonomy` |
| 6 | Report | `generate_comparison_csv` + `export_narrative` |

## References

- Braun & Clarke (2006) — six-phase thematic analysis
- Grootendorst (2022) — BERTopic
- Carlsen & Ralund (2022) — CALM framework
- Jiang et al. (2019) — PAJAIS taxonomy
