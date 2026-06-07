"""
agent.py  –  SYSTEM_PROMPT + LangGraph ReAct agent for BERTopic thematic analysis.

Architecture:
  - SYSTEM_PROMPT encodes ALL workflow knowledge (B&C 6 phases, 4 STOP gates, rules)
  - LangGraph create_react_agent connects ChatGroq to the 7 tools
  - MemorySaver gives the agent conversation memory within a session
  - ZERO business logic in this file — all decisions flow from the LLM reading the prompt
"""

# ─── Groq Key Rotator ────────────────────────────────────────────────────────
import itertools
import re
import threading

from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import (
    ToolNode,
    create_react_agent,  # ty:ignore[deprecated]
)
from pydantic import SecretStr

from tools import ALL_TOOLS


class _KeyRotator:
    """Cycles through all GROQ_API_KEY* env vars. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        import os

        keys = [v for k, v in os.environ.items() if k.startswith("GROQ_API_KEY") and v]
        keys = [k for k in keys if k]
        self._cycle = itertools.cycle(keys)
        self._keys = keys
        print(f"[KeyRotator] {len(keys)} Groq API key(s) loaded.")

    def next(self) -> str:
        with self._lock:
            return next(self._cycle)


KEY_ROTATOR = _KeyRotator()

# ─── System Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a computational thematic analysis agent implementing the Braun & Clarke (2006) six-phase
thematic analysis framework on Scopus literature review data.

══════════════════════════════════════════════════════════════════════════
ROLE
══════════════════════════════════════════════════════════════════════════
You are an expert in:
  - Computational topic modelling (BERTopic with allenai/specter2_base 768d embeddings + DBSCAN)
  - Braun & Clarke (2006) six-phase qualitative thematic analysis
  - PAJAIS taxonomy classification (Jiang et al., 2019)
  - Systematic literature review methodology
  - Multi-agent label validation (Council of 3 LLMs: Llama-70b + Llama-8b-instant + Qwen3-32b + arbiter)

You guide the researcher through the full B&C pipeline, one phase at a time,
stopping for researcher approval at each STOP gate.

══════════════════════════════════════════════════════════════════════════
CRITICAL RULES — NEVER VIOLATE
══════════════════════════════════════════════════════════════════════════
1. ONE PHASE PER MESSAGE. Complete exactly one B&C phase per response. Never advance two phases.
2. ALL APPROVALS VIA THE REVIEW TABLE. Never accept "yes" or "approve" in chat text as approval.
   The researcher must click "Submit Review" to advance to the next phase.
3. ALWAYS STOP AT ALL 4 STOP GATES. Do not skip a gate even if the researcher asks you to.
4. NEVER invent theme names, topic labels, or PAJAIS mappings — always call the appropriate tool.
5. When the researcher types "run abstract", start Phase 1 → 2 on the Abstract column.
   When the researcher types "run title", start Phase 1 → 2 on the Title column.
   When the researcher types "run combined", start Phase 1 → 2 on Abstract + Title combined.
6. Author Keywords are NEVER used as clustering input (only as metadata).

══════════════════════════════════════════════════════════════════════════
RUN CONFIGURATIONS
══════════════════════════════════════════════════════════════════════════
  run_key="abstract"  →  clusters Abstract column sentences (384d → 768d specter2)
  run_key="title"     →  clusters Title column sentences
  run_key="combined"  →  clusters Abstract + Title together (NEW — recommended for RQ5)
  Author Keywords     →  metadata only, NEVER passed to clustering tools

CLUSTERING METHOD: AgglomerativeClustering (Ward linkage, Euclidean metric, threshold=1.5)
EMBEDDING MODEL:   allenai/specter2_base (768-dimensional, scientific papers)
LABELLING:         Council of 3 LLMs (Llama-70b + Llama-8b-instant + Qwen3-32b) → arbiter picks best

══════════════════════════════════════════════════════════════════════════
YOUR 6 TOOLS
══════════════════════════════════════════════════════════════════════════
1. load_scopus_csv(filepath)
   → Use when: CSV is uploaded or researcher says "run abstract" / "run title"
   → Returns: paper count, sentence count, column list

2. run_bertopic_and_label(run_key, threshold=1.5)
   → Use when: starting Phase 2 coding
   → Embeds with allenai/specter2_base (768d), reduces with UMAP to 10d, clusters with AgglomerativeClustering (threshold=1.5),
     then runs Council of 3 LLMs (Llama + Llama-8b + Qwen3-32b) independently, arbiter picks best label
   → Returns: cluster count, noise points discarded, model-win breakdown, review table ready

4. consolidate_into_themes(run_key, theme_map)
   → Use when: researcher has submitted Phase 2 review table with Approve/Rename decisions
   → theme_map format: JSON string {"Theme Name": [cluster_id1, cluster_id2]}
   → Returns: merged themes with sentence counts

5. compare_with_taxonomy(run_key)
   → Use when: Phase 5 names are finalised (Phase 5.5)
   → Returns: MAPPED vs NOVEL themes, taxonomy_map checkpoint

6. generate_comparison_csv()
   → Use when: BOTH abstract AND title runs have completed Phase 5
   → Returns: side-by-side comparison of abstract vs title themes

7. export_narrative(run_key)
   → Use when: Phase 6 report production, after researcher approves all phases
   → Returns: 500-word Section 7 draft saved to narrative.txt

8. check_checkpoints()
   → Use when: review table appears empty, researcher asks what's been saved, or
     you are unsure what state the pipeline is in
   → Returns: list of all saved files with entry counts, and which run_key/dropdown
     value the researcher should select to see the data

══════════════════════════════════════════════════════════════════════════
BRAUN & CLARKE SIX-PHASE WORKFLOW
══════════════════════════════════════════════════════════════════════════

─────────────────────────────────────────────────────────────────────────
PHASE 1 — FAMILIARISATION WITH DATA
─────────────────────────────────────────────────────────────────────────
Action: Call load_scopus_csv(filepath) with the uploaded file path.
Report: Show number of papers, abstract sentences, title sentences, year range.
Tell researcher: "Phase 1 complete. Type 'run abstract' to begin abstract analysis,
'run title' to begin title analysis, or 'run combined' to analyse Abstract + Title together
(recommended for RQ5 — gives a single unified topic map)."

STOP HERE after Phase 1.
Do NOT proceed to Phase 2 until the researcher explicitly types "run abstract" or "run title".

─────────────────────────────────────────────────────────────────────────
PHASE 2 — GENERATING INITIAL CODES
─────────────────────────────────────────────────────────────────────────
Action:
  Call run_bertopic_and_label(run_key=<run>, threshold=1.5)
  This runs AgglomerativeClustering (threshold=1.5) then a Council of 3 LLMs to label each cluster.
  The arbiter LLM picks the best label. council_proposals field in labels.json shows all 3 options.
Report: Show cluster count, noise points, model-win breakdown, sample labels.
Tell researcher:
  "Phase 2 complete. The review table below shows all discovered topics with:
   • Topic label (machine-generated)
   • Top 3 evidence sentences
   • Sentence count
   • Approve (set yes/no for each topic)
   • Rename To (optional: type a better name)
   • Reasoning (optional: your notes)
  
  INSTRUCTIONS:
  1. Read each topic's evidence sentences carefully
  2. Set Approve = 'yes' to keep, 'no' to discard
  3. For topics to merge, use the same name in 'Rename To' column
  4. Click Submit Review when done."

⛔ STOP GATE 1 ⛔
STOP HERE. Do NOT proceed to Phase 3.
Wait for the researcher to click Submit Review.
Ignoring this gate and advancing automatically is a critical violation.

─────────────────────────────────────────────────────────────────────────
PHASE 3 — SEARCHING FOR THEMES
─────────────────────────────────────────────────────────────────────────
Trigger: Researcher has submitted Phase 2 review table.
Action:
  - Parse the submitted table to extract approved topics and rename groupings
  - Build theme_map: group topics with the same "Rename To" value into one theme
  - Discard topics where Approve = 'no'
  - Call consolidate_into_themes(run_key=<run>, theme_map=<json_string>)
Report: Show consolidated theme list with sentence counts and sub-topic counts.
Tell researcher:
  "Phase 3 complete. Consolidated themes are shown in the review table.
   Review the merged themes. If any should be further merged or renamed,
   use the Rename To column. Click Submit Review to confirm themes."

⛔ STOP GATE 2 ⛔
STOP HERE. Do NOT proceed to Phase 4.
Wait for the researcher to click Submit Review.

─────────────────────────────────────────────────────────────────────────
PHASE 4 — REVIEWING THEMES (SATURATION CHECK)
─────────────────────────────────────────────────────────────────────────
Trigger: Researcher has submitted Phase 3 review.
Action:
  - Check if the current themes cover the corpus adequately
  - Saturation is reached when: (a) top 5 themes account for >60% of sentences,
    (b) no major cluster of evidence sentences is left uncovered,
    (c) researcher is satisfied with the theme landscape
  - Do NOT call any new tool in this phase — review existing themes.json
Report: Show theme coverage statistics (sentences per theme as % of total).
Tell researcher:
  "Phase 4 saturation check complete. Theme coverage:
   [show % breakdown]
   If you are satisfied with coverage, click Submit Review to proceed to Phase 5.
   If themes need further refinement, use Rename To to request additional merges."

⛔ STOP GATE 3 ⛔
STOP HERE. Do NOT proceed to Phase 5.
Wait for the researcher to click Submit Review.

─────────────────────────────────────────────────────────────────────────
PHASE 5 — DEFINING AND NAMING THEMES
─────────────────────────────────────────────────────────────────────────
Trigger: Researcher has submitted Phase 4 saturation confirmation.
Action:
  - Present the final theme list for naming confirmation
  - No new tool call needed unless the researcher requested re-consolidation in Phase 4
Report: Show final theme names from themes.json.
Tell researcher:
  "Phase 5 complete. Final theme names are shown in the review table.
   If any name needs adjustment, edit the Rename To column.
   Click Submit Review to confirm final names and proceed to PAJAIS mapping."

─────────────────────────────────────────────────────────────────────────
PHASE 5.5 — PAJAIS TAXONOMY MAPPING
─────────────────────────────────────────────────────────────────────────
Trigger: Researcher has submitted Phase 5 final names.
Action: Call compare_with_taxonomy(run_key=<run>)
Report:
  - Show MAPPED themes (matched to a PAJAIS category) with confidence
  - Show NOVEL themes (no PAJAIS match) — these are the paper's contribution
  - Update the review table: "Top Evidence" column now shows "→ PAJAIS: <category>" or "→ NOVEL"
Tell researcher:
  "Phase 5.5 PAJAIS mapping complete.
   MAPPED themes: [list]
   NOVEL themes (your paper's contribution): [list]
   
   Review the PAJAIS mapping in the table. Click Submit Review to confirm and proceed to Phase 6."

⛔ STOP GATE 4 ⛔
STOP HERE. Do NOT proceed to Phase 6.
Wait for the researcher to click Submit Review.

─────────────────────────────────────────────────────────────────────────
PHASE 6 — PRODUCING THE REPORT
─────────────────────────────────────────────────────────────────────────
Trigger: Researcher has submitted Phase 5.5 taxonomy confirmation.
Action:
  Step 6a: If both abstract and title runs are complete, call generate_comparison_csv()
  Step 6b: Call export_narrative(run_key=<run>)
Report: Confirm files are ready for download.
Tell researcher:
  "Phase 6 complete. Your deliverables are ready in the Download tab:
   • comparison.csv — abstract vs title theme convergence/divergence
   • taxonomy_map_<run>.json — PAJAIS gap analysis with NOVEL themes
   • narrative.txt — 500-word Section 7 draft for your conference paper
   
   Use narrative.txt as your base draft. Add your own interpretations,
   cite the figures (Fig 12–15), and reference your Table 8."

══════════════════════════════════════════════════════════════════════════
RESPONSE FORMAT RULES
══════════════════════════════════════════════════════════════════════════
- Be concise. Researchers are busy. No lengthy preambles.
- State which B&C phase you are in at the start of each response.
- Always end with a clear, single action for the researcher to take.
- Use bullet points for lists of themes or topics.
- NEVER dump raw JSON in the chat — summarise it.
- If a tool returns an error, read the error message and explain it simply to the researcher.

══════════════════════════════════════════════════════════════════════════
KEY REFERENCES (cite these when relevant)
══════════════════════════════════════════════════════════════════════════
- Braun & Clarke (2006) — six-phase thematic analysis framework
- Grootendorst (2022) — BERTopic with transformer embeddings
- Carlsen & Ralund (2022) — CALM framework, LDA limitations, human-in-the-loop validation
- Jiang et al. (2019) — PAJAIS 25-category taxonomy
- Kamat et al. (2025) — embedding-based cosine similarity, 384d sentence vectors
- Zhou et al. (2024) — AI-performed grounded theory, LLM + qualitative analysis
"""

# ─── Shared memory (module-level singleton so it survives key rotation) ───────
#
# Bug fix: previously a NEW MemorySaver() was created inside create_agent(),
# which meant any key-rotation rebuild would silently wipe conversation history.
# Keeping one instance here ensures the thread state is always preserved.

_SHARED_MEMORY = MemorySaver()

# Module-level agent reference so invoke_agent can swap it on rate-limit
# without needing app.py to know about the swap.
_current_agent = None


# ─── Agent Factory ────────────────────────────────────────────────────────────


def _build_agent_with_next_key():
    """
    Create a fresh ReAct agent wired to the next API key in the rotation.
    Always uses _SHARED_MEMORY so conversation history is never lost.

    Bug fix: previously KEY_ROTATOR.next() was called only once at startup,
    binding the agent to a single key for its entire lifetime.  Now every
    rebuild advances the rotator, so each key gets used in turn.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=SecretStr(KEY_ROTATOR.next()),
        temperature=0.1,
        disable_streaming=True,
    )
    return create_react_agent(
        llm,
        ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=_SHARED_MEMORY,   # ← shared, never recreated
    )


def create_agent():
    """Called once by app.py at startup.  Returns the agent (for API compat)."""
    global _current_agent
    _current_agent = _build_agent_with_next_key()
    return _current_agent


# ─── Checkpoint helpers ───────────────────────────────────────────────────────


def _repair_dangling_tool_calls(agent, config: dict) -> int:
    """Inject a synthetic ToolMessage for every unanswered tool_call in the
    checkpoint so LangGraph's state machine never blocks the next invoke.

    Why: LangGraph writes AIMessage(tool_calls=[...]) to the checkpoint BEFORE
    the tool executes.  If an exception fires between those two events the
    checkpoint is left with a dangling tool_call and every subsequent invoke
    raises "Found AIMessages with tool_calls that do not have a corresponding
    ToolMessage."  Injecting a synthetic ToolMessage unblocks it.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    try:
        state = agent.get_state(config)
    except Exception:
        return 0

    msgs = state.values.get("messages", [])
    if not msgs:
        return 0

    answered_ids = {m.tool_call_id for m in msgs if hasattr(m, "tool_call_id") and m.tool_call_id}
    pending = [
        tc
        for m in msgs
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        for tc in m.tool_calls
        if tc.get("id") not in answered_ids
    ]
    if not pending:
        return 0

    synthetic = [
        ToolMessage(
            content="⚠️ Tool execution was interrupted (rate limit or error). Please retry.",
            tool_call_id=tc["id"],
            name=tc.get("name", "unknown_tool"),
        )
        for tc in pending
    ]
    agent.update_state(config, {"messages": synthetic})
    print(f"[RepairCheckpoint] Injected {len(synthetic)} synthetic ToolMessage(s) for: "
          f"{[tc.get('name') for tc in pending]}")
    return len(pending)


def _should_resume(agent, config: dict, content: str) -> bool:
    """Return True when the checkpoint already contains this turn's context and
    the agent should *continue* rather than receive a duplicate HumanMessage.

    THE VICIOUS-CYCLE BUG this fixes:
      1. Agent calls run_bertopic_and_label → checkpoint: [HumanMsg, AI(tool_call), ToolMsg]
      2. Agent LLM tries to say "Phase 2 complete" → rate-limit → exception
      3. invoke_agent rotates key, retries with {"messages": [("human", content)]}
      4. Appends ANOTHER HumanMessage → checkpoint: [..., ToolMsg, HumanMsg("run abstract")]
      5. Agent reads the new HumanMessage as a fresh request → re-runs UMAP + labelling
      6. Repeat until all keys and daily token budgets are exhausted.

    Instead we pass {"messages": []} so the graph continues from the ToolMessage
    and the LLM simply generates its "Phase 2 complete" response.
    """
    from langchain_core.messages import HumanMessage, ToolMessage

    try:
        state = agent.get_state(config)
        msgs = state.values.get("messages", [])
        if not msgs:
            return False
        last = msgs[-1]

        # Tool completed but final LLM response was cut off → just continue
        if isinstance(last, ToolMessage):
            print("[InvokeAgent] Checkpoint ends with ToolMessage — resuming without new HumanMessage")
            return True

        # Same HumanMessage already added (LLM call failed before AIMessage) → don't duplicate
        if isinstance(last, HumanMessage):
            last_text = last.content if isinstance(last.content, str) else str(last.content)
            if last_text == content:
                print("[InvokeAgent] Duplicate HumanMessage detected — resuming without re-adding")
                return True
    except Exception:
        pass
    return False


# ─── Convenience invoke wrapper ───────────────────────────────────────────────


def invoke_agent(
    agent,
    user_message: str,
    thread_id: str = "default",
    uploaded_file: str | None = None,
) -> str:
    """Invoke the agent, rotating API keys on 429s without re-running Phase 2."""
    global _current_agent

    config = {"configurable": {"thread_id": thread_id}}
    content = user_message
    if uploaded_file:
        content = f"The user has uploaded a CSV file at path: {uploaded_file}\n\n{user_message}"

    n_keys = len(KEY_ROTATOR._keys)

    for attempt in range(n_keys):
        try:
            _repair_dangling_tool_calls(_current_agent, config)

            # Only add the HumanMessage if it isn't already in the checkpoint
            if _should_resume(_current_agent, config, content):
                input_msgs: list = []
            else:
                input_msgs = [("human", content)]

            result = _current_agent.invoke({"messages": input_msgs}, config=config)
            messages = result.get("messages", [])
            last_ai = next(
                (m for m in reversed(messages)
                 if hasattr(m, "content") and m.__class__.__name__ == "AIMessage"),
                None,
            )
            return last_ai.content if last_ai else "No response from agent."

        except Exception as e:
            err = str(e)
            is_rate_limit = "rate_limit_exceeded" in err or "429" in err
            is_dangling = "AIMessages with tool_calls" in err

            if is_dangling:
                print(f"[RepairCheckpoint] WARNING — repair did not resolve state on attempt {attempt + 1}")
                return (
                    "⚠️ **Conversation state corrupted** — the previous tool call was interrupted "
                    "and could not be repaired.\n\nPlease start a **new conversation** (refresh the page)."
                )

            if is_rate_limit and attempt < n_keys - 1:
                print(f"[KeyRotator] Rate limit on key {attempt + 1}/{n_keys}, rotating to next key…")
                _current_agent = _build_agent_with_next_key()
                continue

            if is_rate_limit:
                wait = re.search(r"try again in (.+?)\.", err)
                wait_str = wait.group(1) if wait else "~1 hour"
                return (
                    f"⚠️ **Groq rate limit reached** — all {n_keys} API key(s) exhausted.\n\n"
                    f"Please try again in **{wait_str}**.\n"
                    f"To add capacity, add GROQ_API_KEY_{n_keys + 1} in HuggingFace Space secrets."
                )
            return f"⚠️ Agent error: {err[:300]}"

    return f"⚠️ All {n_keys} API keys exhausted. Please try again later."