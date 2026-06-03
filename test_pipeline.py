#!/usr/bin/env python3
"""
test_pipeline.py — automated smoke test for BERTopic Agentic AI on HF Spaces.
No drag-and-drop. No typing 'run abstract'. Just run this and go get a coffee.

Usage:
  # Quick sanity check — discover what API endpoints exist:
  python test_pipeline.py --show-api --hf-token hf_xxx

  # Full pipeline smoke test (needs your journal CSV):
  python test_pipeline.py --csv journal.csv --hf-token hf_xxx

  # Quick smoke test with a generated synthetic CSV (no real data needed):
  python test_pipeline.py --synth --hf-token hf_xxx

  # Just tail the Space logs in real time (readonly token is fine):
  python test_pipeline.py --logs --log-token hf_yyy

  # Check Space status / build stage:
  python test_pipeline.py --status --log-token hf_yyy

Requirements:  pip install gradio_client huggingface_hub requests
"""

import argparse
import csv
import json
import random
import sys
import tempfile
import time
from pathlib import Path

try:
    import gradio_client.utils as _gcu

    _orig_parse = _gcu._json_schema_to_python_type

    def _safe_parse(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_parse(schema, defs)

    _gcu._json_schema_to_python_type = _safe_parse  # ty:ignore[invalid-assignment]
except (ImportError, AttributeError):
    pass

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_SPACE = "RuchirKadam1729/bertopic-agentic-ai"
TEST_THREAD = f"smoke-test-{int(time.time())}"  # unique thread per run
TIMEOUT_PHASE2 = 600  # 10 min max for Phase 2 (embedding + UMAP + cluster + label)

# ─── Arg parsing ─────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="BERTopic app smoke tester")
    p.add_argument("--csv", help="Path to your Scopus CSV")
    p.add_argument(
        "--synth",
        action="store_true",
        help="Generate a synthetic CSV (100 papers, 5 topics) and use that",
    )
    p.add_argument("--hf-token", help="HF read+write token (needed to run the app)")
    p.add_argument("--log-token", help="HF readonly token (for --logs and --status)")
    p.add_argument("--space-id", default=DEFAULT_SPACE)
    p.add_argument(
        "--run-key", default="abstract", choices=["abstract", "title", "combined"]
    )
    p.add_argument(
        "--show-api", action="store_true", help="Print available API endpoints and exit"
    )
    p.add_argument(
        "--logs", action="store_true", help="Stream Space container logs to stdout"
    )
    p.add_argument(
        "--status", action="store_true", help="Print Space build/runtime status"
    )
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def ok(msg):
    print(f"  {GREEN}✓{RESET}  {msg}")


def fail(msg):
    print(f"  {RED}✗{RESET}  {msg}")


def warn(msg):
    print(f"  {YELLOW}!{RESET}  {msg}")


def info(msg):
    print(f"     {msg}")


def step(n, msg):
    print(f"\n{BOLD}[{n}]{RESET} {msg}")


def elapsed(t0):
    return f"{time.time() - t0:.1f}s"


# ─── Synthetic CSV ────────────────────────────────────────────────────────────

TOPIC_TEMPLATES = {
    "E-Commerce and Consumer Behavior": [
        "This study examines {aspect} in the context of online consumer purchasing decisions.",
        "We investigate how {tech} influences trust and loyalty in e-commerce platforms.",
        "The findings suggest that {factor} significantly mediates {outcome} for digital shoppers.",
        "Our empirical analysis of {n} consumers reveals important insights about {behavior}.",
        "Results indicate that {aspect} is a critical determinant of repeat purchase intentions.",
    ],
    "Artificial Intelligence and Machine Learning": [
        "This paper proposes a novel {method} for {task} using deep neural networks.",
        "We apply machine learning techniques to analyse {domain} at scale.",
        "The proposed {framework} achieves state-of-the-art performance on {benchmark}.",
        "Our experiments demonstrate that {approach} outperforms existing baselines in {metric}.",
        "We present a {model} that combines {tech1} and {tech2} for improved {outcome}.",
    ],
    "Security and Privacy in IS": [
        "This research investigates {threat} vulnerabilities in enterprise information systems.",
        "We propose a {protocol} for protecting user data against {attack} attacks.",
        "Our framework addresses privacy concerns related to {tech} in organisational contexts.",
        "Findings from {n} firms indicate that {practice} reduces security breach probability.",
        "The study develops a {model} for assessing cyber risk in {sector} organisations.",
    ],
    "Mobile Computing and Apps": [
        "This study explores user adoption of {service} on mobile devices across {context}.",
        "We examine how {feature} affects engagement with mobile {type} applications.",
        "A survey of {n} mobile users reveals that {factor} drives sustained app usage.",
        "Our analysis shows that {aspect} moderates the relationship between usability and adoption.",
        "Results confirm that {design} principles improve task completion rates in mobile {domain}.",
    ],
    "Knowledge Management Systems": [
        "This paper examines how {tool} supports knowledge sharing in distributed organisations.",
        "We investigate the role of {factor} in facilitating tacit knowledge transfer.",
        "Our case study of {sector} firms demonstrates the impact of {system} on innovation.",
        "Findings suggest that {practice} enhances knowledge retention and organisational learning.",
        "We develop a {framework} for measuring knowledge management maturity in {context}.",
    ],
}

FILLERS = {
    "aspect": [
        "perceived usefulness",
        "interface design",
        "social influence",
        "personalisation",
    ],
    "tech": [
        "AI recommendations",
        "blockchain",
        "augmented reality",
        "chatbots",
        "IoT",
    ],
    "factor": ["trust", "ease of use", "perceived risk", "social norms", "habit"],
    "outcome": [
        "satisfaction",
        "loyalty",
        "engagement",
        "conversion rate",
        "retention",
    ],
    "n": ["350", "420", "512", "278", "634", "189"],
    "behavior": [
        "impulsive buying",
        "review generation",
        "subscription renewal",
        "cart abandonment",
    ],
    "method": [
        "convolutional neural network",
        "transformer model",
        "hybrid approach",
        "ensemble",
    ],
    "task": [
        "sentiment analysis",
        "fraud detection",
        "demand forecasting",
        "anomaly detection",
    ],
    "domain": [
        "customer reviews",
        "financial transactions",
        "social media data",
        "log files",
    ],
    "framework": [
        "multi-layer architecture",
        "federated learning approach",
        "attention mechanism",
    ],
    "benchmark": [
        "standard IS datasets",
        "cross-industry evaluations",
        "held-out test sets",
    ],
    "approach": ["the proposed system", "our integrated model", "the hybrid algorithm"],
    "metric": ["accuracy", "F1 score", "AUC-ROC", "precision-recall"],
    "model": [
        "predictive model",
        "classification framework",
        "decision support system",
    ],
    "tech1": [
        "natural language processing",
        "graph neural networks",
        "transfer learning",
    ],
    "tech2": ["reinforcement learning", "attention mechanisms", "contrastive learning"],
    "threat": ["SQL injection", "phishing", "ransomware", "insider threat", "zero-day"],
    "protocol": [
        "end-to-end encryption protocol",
        "zero-trust framework",
        "RBAC scheme",
    ],
    "attack": ["man-in-the-middle", "credential stuffing", "social engineering"],
    "practice": [
        "security awareness training",
        "multi-factor authentication",
        "patching",
    ],
    "sector": ["financial", "healthcare", "retail", "manufacturing", "public"],
    "service": ["mobile banking", "health tracking", "food delivery", "ride-sharing"],
    "type": ["productivity", "entertainment", "social", "fitness", "finance"],
    "design": ["minimalist", "gesture-based", "voice-first", "adaptive"],
    "context": [
        "emerging markets",
        "enterprise settings",
        "SMEs",
        "educational institutions",
    ],
    "feature": ["push notifications", "gamification", "offline mode", "dark mode"],
    "tool": ["enterprise wikis", "document management systems", "AI-assisted search"],
    "system": ["integrated KMS", "AI-powered platform", "collaborative intranet"],
    "practice": ["communities of practice", "mentoring programmes", "knowledge audits"],
}


def fill(template):
    result = template
    for key, options in FILLERS.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(options), 1)
    return result


def make_synthetic_csv(path: str, papers_per_topic: int = 20):
    """Generate a minimal but realistic Scopus-style CSV for smoke testing."""
    random.seed(42)
    rows = []
    sr = 1
    for topic, templates in TOPIC_TEMPLATES.items():
        for i in range(papers_per_topic):
            # Build a 4-sentence abstract from the templates
            abstract_sents = [fill(random.choice(templates)) for _ in range(4)]
            abstract = " ".join(abstract_sents)
            rows.append(
                {
                    "Sr No": sr,
                    "Authors": f"Author{sr}A, Author{sr}B",
                    "Title": f"{topic}: A Study of {fill('{aspect}')} ({2010 + (sr % 14)})",
                    "Abstract": abstract,
                    "Author Keywords": topic.lower().replace(" ", "; "),
                    "Cited by": random.randint(0, 200),
                    "Source title": "Electronic Markets",
                    "Year": 2010 + (sr % 14),
                }
            )
            sr += 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Generated synthetic CSV: {path}")
    print(f"  {len(rows)} papers across {len(TOPIC_TEMPLATES)} topics")
    return path


# ─── Status ───────────────────────────────────────────────────────────────────


def check_status(space_id: str, token: str):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"\n{'─' * 55}")
    print(f"  Space: {space_id}")
    print(f"{'─' * 55}")
    try:
        runtime = api.get_space_runtime(space_id)
        stage = str(runtime.stage)
        colour = GREEN if "RUNNING" in stage else (RED if "ERROR" in stage else YELLOW)
        print(f"  Stage:    {colour}{stage}{RESET}")
        if hasattr(runtime, "sdk"):
            print(f"  SDK:      {runtime.sdk}")
        if hasattr(runtime, "hardware"):
            print(f"  Hardware: {runtime.hardware}")
    except Exception as e:
        warn(f"Could not read runtime: {e}")
    try:
        info_obj = api.space_info(space_id)
        if info_obj.last_modified:
            print(f"  Modified: {info_obj.last_modified}")
        if info_obj.sha:
            print(f"  SHA:      {info_obj.sha[:12]}")
    except Exception as e:
        warn(f"Could not read space info: {e}")
    print()


# ─── Log streaming ────────────────────────────────────────────────────────────


def stream_logs(space_id: str, token: str):
    import requests

    url = f"https://huggingface.co/api/spaces/{space_id}/logs"
    headers = {"Authorization": f"Bearer {token}"}
    print(f"Streaming logs for {space_id}  (Ctrl+C to stop)...\n")
    try:
        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            if r.status_code == 403:
                fail(
                    "403 — token can't read logs (needs at least read access to the Space repo)"
                )
                return
            if r.status_code != 200:
                fail(f"HTTP {r.status_code}: {r.text[:200]}")
                return
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                decoded = raw_line.decode("utf-8", errors="replace")
                if decoded.startswith("data: "):
                    try:
                        event = json.loads(decoded[6:])
                        ts = event.get("timestamp", "")
                        msg = event.get("data", "")
                        level = event.get("level", "info").upper()
                        colour = (
                            RED
                            if level == "ERROR"
                            else (YELLOW if level == "WARNING" else "")
                        )
                        print(f"[{ts}] {colour}{level}{RESET}: {msg}")
                    except json.JSONDecodeError:
                        print(decoded)
                else:
                    print(decoded)
    except KeyboardInterrupt:
        print("\nLog stream stopped.")


# ─── API discovery ────────────────────────────────────────────────────────────


def show_api(space_id: str, token: str):
    from gradio_client import Client

    print(f"Connecting to {space_id} ...\n")
    client = Client(space_id, hf_token=token)
    print("Available API endpoints:\n")
    client.view_api()


# ─── Full pipeline smoke test ─────────────────────────────────────────────────


def smoke_test(space_id: str, csv_path: str, hf_token: str, run_key: str):
    from gradio_client import Client, handle_file

    print(f"\n{'─' * 55}")
    print(f"  {BOLD}BERTopic Agentic AI — Smoke Test{RESET}")
    print(f"{'─' * 55}")
    print(f"  Space:   {space_id}")
    print(f"  CSV:     {csv_path}")
    print(f"  Run key: {run_key}")
    print(f"  Thread:  {TEST_THREAD}")
    print()

    t_total = time.time()

    # ── 1. Connect ─────────────────────────────────────────────────────────
    step(1, "Connecting to Space (may take 30s if sleeping)...")
    t = time.time()
    try:
        client = Client(space_id, hf_token=hf_token)
        ok(f"Connected — {elapsed(t)}")
    except Exception as e:
        fail(f"Connection failed: {e}")
        return False

    # ── 2. Upload CSV ──────────────────────────────────────────────────────
    step(2, "Uploading CSV → Phase 1 (Familiarisation)...")
    t = time.time()
    try:
        result = client.predict(
            handle_file(csv_path),
            [],  # chatbot history (empty at start)
            TEST_THREAD,  # thread_id
            api_name="/on_csv_upload",
        )
        history, _ = result
        uploaded_path = None  # gr.State not in API response
        ok(f"Upload done — {elapsed(t)}")

        last_msg = history[-1]["content"] if history else "(no message)"
        info(f"Agent: {last_msg[:250]}")
        info(f"Saved to: {uploaded_path}")
    except Exception as e:
        fail(f"CSV upload failed: {e}")
        info("Try --show-api to check endpoint names match your Gradio version")
        return False

    # ── 3. run abstract / title / combined ────────────────────────────────
    step(3, f"Sending 'run {run_key}' → Phase 2 (embed + UMAP + cluster + label)...")
    info(
        "  Expected time: 3–6 min. Labelling now batches at 15 topics to stay under Groq TPM."
    )
    t = time.time()
    try:
        result = client.predict(
            f"run {run_key}",
            history,
            uploaded_path,
            TEST_THREAD,
            api_name="/on_send",
        )
        history, *_ = result
        dur = time.time() - t
        ok(f"Phase 2 done — {dur:.0f}s ({dur / 60:.1f} min)")

        last_msg = history[-1]["content"] if history else "(no message)"
        info(f"Agent:\n  {last_msg[:500]}")
    except Exception as e:
        fail(f"Phase 2 failed: {e}")
        return False

    # ── 4. Check review table ──────────────────────────────────────────────
    step(4, "Fetching review table — checking labels were generated...")
    t = time.time()
    try:
        raw = client.predict(run_key, api_name="/on_refresh_table")

        # Gradio 5 returns a dict with {"headers": [...], "data": [[...], ...]}
        if isinstance(raw, dict) and "data" in raw:
            rows = raw["data"]
        elif hasattr(raw, "values"):  # pandas DataFrame
            rows = raw.values.tolist()
        elif isinstance(raw, list):
            rows = raw
        else:
            rows = []

        total = len(rows)
        labeled = sum(
            1
            for r in rows
            if r
            and len(r) > 1
            and r[1]
            and str(r[1]).strip()
            and not str(r[1]).startswith("Topic ")
        )
        unlabeled = total - labeled

        info(f"Topics in table:   {total}")
        info(f"Labeled (by LLM):  {labeled}")
        info(f"Unlabeled (bad):   {unlabeled}")

        if total == 0:
            fail("Table is empty — Phase 2 did not complete at all")
            return False
        elif unlabeled == total:
            fail("Every topic still shows 'Topic N' placeholder — labelling broken")
            return False
        elif unlabeled > total * 0.15:
            warn(
                f"{unlabeled}/{total} topics are unlabeled (>15% fallback rate, check Groq rate limits)"
            )
        else:
            ok(f"Labels generated correctly ({labeled}/{total} topics labelled)")

        print()
        info("Sample labels from table:")
        for row in rows[:5] if rows else []:
            if row and len(row) > 1:
                label = str(row[1])[:60]
                sents = str(row[3]) if len(row) > 3 else "?"
                info(f"  #{str(row[0]):>3}  {label:<60}  ({sents} sentences)")

    except Exception as e:
        fail(f"Table fetch failed: {e}")
        return False

    # ── 5. Summary ────────────────────────────────────────────────────────
    total_time = time.time() - t_total
    step(5, f"Done — total time {total_time:.0f}s ({total_time / 60:.1f} min)")
    ok("Smoke test PASSED") if labeled > 0 else fail("Smoke test FAILED")
    return labeled > 0


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    token = args.log_token or args.hf_token

    if args.status:
        if not token:
            print("Need --log-token or --hf-token")
            sys.exit(1)
        check_status(args.space_id, token)
        return

    if args.logs:
        if not token:
            print("Need --log-token or --hf-token")
            sys.exit(1)
        stream_logs(args.space_id, token)
        return

    if args.show_api:
        if not args.hf_token:
            print("Need --hf-token to connect")
            sys.exit(1)
        show_api(args.space_id, args.hf_token)
        return

    # ── Smoke test mode ────────────────────────────────────────────────────
    if not args.hf_token:
        print("Need --hf-token to run the smoke test")
        sys.exit(1)

    if args.synth:
        tmp = tempfile.NamedTemporaryFile(
            suffix="_ElectronicMarkets_TopicModelling_Export.csv", delete=False
        )
        csv_path = make_synthetic_csv(tmp.name)
    elif args.csv:
        if not Path(args.csv).exists():
            print(f"CSV not found: {args.csv}")
            sys.exit(1)
        csv_path = args.csv
    else:
        print("Need either --csv path/to/file.csv  or  --synth")
        sys.exit(1)

    success = smoke_test(
        space_id=args.space_id,
        csv_path=csv_path,
        hf_token=args.hf_token,
        run_key=args.run_key,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
