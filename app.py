"""
app.py  –  Gradio web interface for the BERTopic Agentic AI application.

Architecture:
  - Section ① DATA INPUT:   CSV upload → auto-triggers agent Phase 1
  - Section ② CONVERSATION: Chatbot + text input + Send button
  - Section ③ RESULTS:      Review table (8 editable columns) + Charts + Downloads

Rules:
  - ZERO business logic in this file
  - All decisions are made by the agent (agent.py / tools.py)
  - app.py only handles UI events, routing messages to the agent, and rendering outputs
"""
# ── PATCHES — must be first, before any other import ──────────────────────
import gradio_client.utils as _gcu

# Patch 1: handle non-dict schemas (LangChain tool schema incompatibility)
_orig_parse = _gcu._json_schema_to_python_type
def _safe_parse(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_parse(schema, defs)
_gcu._json_schema_to_python_type = _safe_parse

# Patch 2: fix localhost check failing inside Docker
import gradio.networking as _gnet
_gnet.is_localhost_accessible = lambda: True
# ── END PATCHES ────────────────────────────────────────────────────────────
import os
import json
import gradio as gr
from pathlib import Path
from agent import create_agent, invoke_agent

# ─── Constants ────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

REVIEW_TABLE_HEADERS = [
    "#",
    "Topic Label",
    "Top Evidence",
    "Sentences",
    "Papers",
    "Approve",
    "Rename To",
    "Reasoning",
]

PHASE_LABELS = [
    "① Load",
    "② Codes",
    "③ Themes",
    "④ Saturation",
    "⑤ Names",
    "⑤½ PAJAIS",
    "⑥ Report",
]

CHART_OPTIONS = {
    "Intertopic distance map": "chart_map",
    "Topic sizes (bar chart)": "chart_bar",
    "Topic similarity heatmap": "chart_heat",
}

# ─── Agent Singleton ──────────────────────────────────────────────────────────

agent = create_agent()
THREAD_ID = "bertopic-session-1"

# ─── Helper: Phase Progress HTML ─────────────────────────────────────────────


def get_phase_progress_html(run_key: str = "abstract") -> str:
    checkpoints = {
        "① Load": CHECKPOINT_DIR / "stats.json",
        "② Codes": CHECKPOINT_DIR / f"labels_{run_key}.json",
        "③ Themes": CHECKPOINT_DIR / f"themes_{run_key}.json",
        "⑤½ PAJAIS": CHECKPOINT_DIR / f"taxonomy_map_{run_key}.json",
        "⑥ Report": CHECKPOINT_DIR / "comparison.csv",
    }
    phases = PHASE_LABELS
    statuses = list(
        map(
            lambda p: (
                "done" if checkpoints.get(p, Path("__none__")).exists() else "pending"
            ),
            phases,
        )
    )

    def phase_chip(idx):
        label, status = phases[idx], statuses[idx]
        bg = "#E1F5EE" if status == "done" else "#F1EFE8"
        color = "#0F6E56" if status == "done" else "#5F5E5A"
        icon = "✓" if status == "done" else "○"
        return f'<span style="background:{bg};color:{color};border-radius:12px;padding:3px 10px;font-size:12px;font-weight:500;white-space:nowrap">{icon} {label}</span>'

    chips = "".join(list(map(phase_chip, range(len(phases)))))
    return f'<div style="display:flex;flex-wrap:wrap;gap:6px;padding:8px 0 4px">{chips}</div>'


# ─── Helper: Load Review Table ────────────────────────────────────────────────


def load_review_table(run_key: str = "abstract") -> list:
    """Load review table from the most advanced checkpoint available."""
    tax_path = CHECKPOINT_DIR / f"taxonomy_map_{run_key}.json"
    themes_path = CHECKPOINT_DIR / f"themes_{run_key}.json"
    labels_path = CHECKPOINT_DIR / f"labels_{run_key}.json"
    summaries_path = CHECKPOINT_DIR / f"summaries_{run_key}.json"

    if tax_path.exists():
        data = json.loads(tax_path.read_text())
        return list(
            map(
                lambda i_t: [
                    i_t[0] + 1,
                    i_t[1].get("theme_name", ""),
                    f"→ {i_t[1].get('pajais_match', 'NOVEL')} | {i_t[1].get('reasoning', '')}",
                    i_t[1].get("total_sentences", ""),
                    i_t[1].get("sub_topics", ""),
                    "yes",
                    "",
                    i_t[1].get("reasoning", ""),
                ],
                enumerate(data),
            )
        )

    if themes_path.exists():
        data = json.loads(themes_path.read_text())
        return list(
            map(
                lambda i_t: [
                    i_t[0] + 1,
                    i_t[1].get("theme_name", ""),
                    "; ".join(i_t[1].get("representative_sentences", [])[:1]),
                    i_t[1].get("total_sentences", ""),
                    i_t[1].get("sub_topics", ""),
                    "yes",
                    "",
                    "",
                ],
                enumerate(data),
            )
        )

    if labels_path.exists():
        data = json.loads(labels_path.read_text())
        return list(
            map(
                lambda i_t: [
                    i_t[0] + 1,
                    i_t[1].get("label", f"Topic {i_t[1].get('cluster_id', i_t[0])}"),
                    (
                        i_t[1].get("top_sentences", [""])[0]
                        if i_t[1].get("top_sentences")
                        else ""
                    ),
                    i_t[1].get("size", ""),
                    "",
                    "",
                    "",
                    i_t[1].get("reasoning", ""),
                ],
                enumerate(data[:100]),
            )
        )

    if summaries_path.exists():
        data = json.loads(summaries_path.read_text())
        return list(
            map(
                lambda i_t: [
                    i_t[0] + 1,
                    f"Topic {i_t[1].get('cluster_id', i_t[0])}",
                    (
                        i_t[1].get("top_sentences", [""])[0]
                        if i_t[1].get("top_sentences")
                        else ""
                    ),
                    i_t[1].get("size", ""),
                    "",
                    "",
                    "",
                    "",
                ],
                enumerate(data[:100]),
            )
        )

    return []


# ─── Helper: Get Chart HTML ────────────────────────────────────────────────────


def get_chart_html(chart_name: str, run_key: str = "abstract") -> str:
    key = CHART_OPTIONS.get(chart_name, "chart_bar")
    path = CHECKPOINT_DIR / f"{key}_{run_key}.html"
    if not path.exists():
        return "<p style='color:var(--color-text-secondary);padding:20px'>Chart not yet generated. Run topic discovery first.</p>"
    content = path.read_text()
    return f'<iframe srcdoc="{content.replace(chr(34), chr(39))}" style="width:100%;height:450px;border:none;border-radius:8px"></iframe>'


# ─── Helper: Get Download Files ───────────────────────────────────────────────


def get_download_files() -> list:
    candidates = [
        CHECKPOINT_DIR / "comparison.csv",
        CHECKPOINT_DIR / "taxonomy_map_abstract.json",
        CHECKPOINT_DIR / "taxonomy_map_title.json",
        CHECKPOINT_DIR / "narrative.txt",
        CHECKPOINT_DIR / "themes_abstract.json",
        CHECKPOINT_DIR / "themes_title.json",
        CHECKPOINT_DIR / "labels_abstract.json",
        CHECKPOINT_DIR / "labels_title.json",
    ]
    return list(map(str, filter(lambda p: p.exists(), candidates)))


# ─── Helper: Parse submitted review table → theme_map JSON ───────────────────


def parse_review_table_to_theme_map(table_data: list) -> str:
    approved = list(
        filter(
            lambda row: str(row[5]).strip().lower() in ("yes", "y", "1", "true"),
            table_data,
        )
    )
    theme_groups = {}
    list(
        map(
            lambda row: theme_groups.setdefault(
                str(row[6]).strip() if str(row[6]).strip() else str(row[1]).strip(), []
            ).append(int(row[0]) - 1),
            approved,
        )
    )
    return json.dumps(theme_groups)


# ─── Gradio Application ───────────────────────────────────────────────────────

with gr.Blocks(title="BERTopic Agentic AI") as app:

    # State
    run_key_state = gr.State("abstract")
    uploaded_path_state = gr.State(None)

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML(
        """
        <div style="padding:16px 0 8px">
          <h1 style="font-size:20px;font-weight:500;margin:0 0 4px">BERTopic Agentic AI</h1>
          <p style="font-size:13px;color:var(--color-text-secondary);margin:0">
            Braun &amp; Clarke (2006) · 6-phase thematic analysis · PAJAIS taxonomy mapping
          </p>
        </div>
    """
    )

    phase_progress = gr.HTML(get_phase_progress_html())

    # ── Section ①: Data Input ────────────────────────────────────────────────
    with gr.Group():
        gr.Markdown("**① Data input**", elem_classes=["section-label"])
        csv_upload = gr.File(
            label="Upload Scopus CSV ([JournalName]_TopicModelling_Export.csv)",
            file_types=[".csv"],
        )

    # ── Section ②: Agent Conversation ───────────────────────────────────────
    with gr.Group():
        gr.Markdown("**② Agent conversation**", elem_classes=["section-label"])
        chatbot = gr.Chatbot(height=420, show_copy_button=True, type="messages")
        with gr.Row():
            user_input = gr.Textbox(
                placeholder='Type "run abstract" to start, or ask any question...',
                show_label=False,
                scale=5,
            )
            send_btn = gr.Button("Send →", scale=1, variant="primary")

    # ── Section ③: Results ──────────────────────────────────────────────────
    with gr.Group():
        gr.Markdown("**③ Results**", elem_classes=["section-label"])
        with gr.Tabs():

            with gr.Tab("Review table"):
                gr.Markdown(
                    "_Edit **Approve** (yes/no), **Rename To** (merge topics with same name), and **Reasoning**. Then click Submit Review._",
                    elem_classes=["hint-text"],
                )
                review_table = gr.Dataframe(
                    headers=REVIEW_TABLE_HEADERS,
                    datatype=[
                        "number",
                        "str",
                        "str",
                        "number",
                        "number",
                        "str",
                        "str",
                        "str",
                    ],
                    interactive=True,
                    wrap=True,
                    row_count=(10, "dynamic"),
                )
                with gr.Row():
                    run_selector = gr.Dropdown(
                        choices=["abstract", "title"],
                        value="abstract",
                        label="Run",
                        scale=1,
                    )
                    refresh_table_btn = gr.Button("Refresh table", scale=1)
                    submit_review_btn = gr.Button(
                        "Submit review →", scale=2, variant="primary"
                    )

            with gr.Tab("Charts"):
                chart_selector = gr.Dropdown(
                    choices=list(CHART_OPTIONS.keys()),
                    value=list(CHART_OPTIONS.keys())[0],
                    label="Select chart",
                )
                chart_run_selector = gr.Dropdown(
                    choices=["abstract", "title"],
                    value="abstract",
                    label="Run",
                )
                chart_display = gr.HTML(
                    "<p style='color:var(--color-text-secondary);padding:20px'>Run topic discovery to see charts.</p>"
                )
                chart_selector.change(
                    fn=get_chart_html,
                    inputs=[chart_selector, chart_run_selector],
                    outputs=chart_display,
                )
                chart_run_selector.change(
                    fn=get_chart_html,
                    inputs=[chart_selector, chart_run_selector],
                    outputs=chart_display,
                )

            with gr.Tab("Downloads"):
                gr.Markdown("Files are generated as you complete each phase.")
                download_files = gr.File(
                    label="Download deliverables",
                    file_count="multiple",
                    interactive=False,
                )
                refresh_downloads_btn = gr.Button("Refresh downloads")

    # ─── Event: CSV Upload ────────────────────────────────────────────────────

    def on_csv_upload(file_obj, history):
        if file_obj is None:
            return history, gr.update(), None

        filepath = file_obj.name
        response = invoke_agent(
            agent,
            f"A CSV file has been uploaded. Please load and analyse it. File path: {filepath}",
            thread_id=THREAD_ID,
        )
        history = history or []
        history.append({"role": "assistant", "content": response})
        return history, get_phase_progress_html(), filepath

    csv_upload.upload(
        fn=on_csv_upload,
        inputs=[csv_upload, chatbot],
        outputs=[chatbot, phase_progress, uploaded_path_state],
    )

    # ─── Event: Send Message ──────────────────────────────────────────────────

    def on_send(user_msg, history, uploaded_path):
        if not user_msg.strip():
            return history, "", get_phase_progress_html()

        history = history or []
        history.append({"role": "user", "content": user_msg})

        response = invoke_agent(
            agent,
            user_msg,
            thread_id=THREAD_ID,
            uploaded_file=uploaded_path,
        )
        history.append({"role": "assistant", "content": response})
        return history, "", get_phase_progress_html()

    send_btn.click(
        fn=on_send,
        inputs=[user_input, chatbot, uploaded_path_state],
        outputs=[chatbot, user_input, phase_progress],
    )
    user_input.submit(
        fn=on_send,
        inputs=[user_input, chatbot, uploaded_path_state],
        outputs=[chatbot, user_input, phase_progress],
    )

    # ─── Event: Refresh Table ─────────────────────────────────────────────────

    def on_refresh_table(run_key):
        return load_review_table(run_key)

    refresh_table_btn.click(
        fn=on_refresh_table,
        inputs=[run_selector],
        outputs=[review_table],
    )

    # ─── Event: Submit Review ─────────────────────────────────────────────────

    def on_submit_review(table_data, run_key, history):
        history = history or []
        theme_map_json = parse_review_table_to_theme_map(table_data)

        message = (
            f"The researcher has submitted the review table for run_key='{run_key}'.\n"
            f"Here are the approved groupings as a theme_map JSON:\n{theme_map_json}\n\n"
            f"Please proceed to the next B&C phase."
        )

        response = invoke_agent(agent, message, thread_id=THREAD_ID)
        history.append({"role": "assistant", "content": response})

        updated_table = load_review_table(run_key)
        return history, updated_table, get_phase_progress_html()

    submit_review_btn.click(
        fn=on_submit_review,
        inputs=[review_table, run_selector, chatbot],
        outputs=[chatbot, review_table, phase_progress],
    )

    # ─── Event: Refresh Downloads ─────────────────────────────────────────────

    refresh_downloads_btn.click(
        fn=get_download_files,
        outputs=[download_files],
    )

    # ─── On Load ─────────────────────────────────────────────────────────────

    app.load(
        fn=lambda: (
            get_phase_progress_html(),
            load_review_table("abstract"),
            get_download_files(),
        ),
        outputs=[phase_progress, review_table, download_files],
    )


# ─── Launch ───────────────────────────────────────────────────────────────────

# app.py — bottom of file
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)