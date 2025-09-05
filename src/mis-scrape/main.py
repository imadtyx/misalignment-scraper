from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import traceback

# Ensure local package imports work when running this script directly
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

# Scrapers
try:
    from scrape.GeminiScraper import GeminiScraper  # type: ignore
except Exception:
    from scrape.GeminiScraper import GeminiScraper  # noqa: F401
try:
    from scrape.ChatGPTScraper import ChatGPTScraper  # type: ignore
    from scrape.ClaudeScraper import ClaudeScraper  # type: ignore
    from scrape.GrokScraper import GrokScraper  # type: ignore
except Exception:
    pass
try:
    from scrape.RedditScraper import RedditScraper  # type: ignore
except Exception:
    pass
try:
    from scrape.GithubScraper import GithubScraper  # type: ignore
except Exception:
    pass

from reproduce.Reproducer import Reproducer  # type: ignore


def detect_platform(url: str) -> str:
    u = url.lower()
    if "gemini.google.com/share" in u:
        return "gemini"
    if "chat.openai.com/share" in u or "chatgpt.openai.com/share" in u:
        return "chatgpt"
    if "claude.ai/share" in u or re.search(r"claude\.ai/chat/", u):
        return "claude"
    if re.search(r"(grok|xai)\.(com|ai)/share/", u):
        return "grok"
    if "reddit.com/r/" in u and "/comments/" in u:
        return "reddit"
    if re.search(r"github\.com/.+/.+/issues/\d+", u):
        return "github"
    return "unknown"


def scrape_to_transcript(url: str) -> Dict[str, Any]:
    plat = detect_platform(url)
    if plat == "gemini":
        s = GeminiScraper(headless=True, timeout_sec=60)
        t = s.scrape(url)
        return t.to_dict()
    if plat == "chatgpt":
        s = ChatGPTScraper(headless=True, timeout_sec=60)
        t = s.scrape(url)
        return t.to_dict()
    if plat == "claude":
        s = ClaudeScraper(headless=True, timeout_sec=60)
        t = s.scrape(url)
        return t.to_dict()
    if plat == "grok":
        s = GrokScraper(headless=True, timeout_sec=60)
        t = s.scrape(url)
        return t.to_dict()
    if plat == "reddit":
        s = RedditScraper()
        t = s.scrape(url)
        return t.to_dict()
    if plat == "github":
        s = GithubScraper()
        t = s.scrape(url)
        return t.to_dict()
    raise ValueError("Unsupported URL; no scraper found")


def to_reproducer_messages(transcript: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a scraped transcript into Reproducer's {"messages":[...]} shape.
    - user/model roles pass through
    - other roles default to user
    """
    out: List[Dict[str, Any]] = []
    for m in transcript.get("messages", []):
        role = (m.get("role") or "user").lower()
        if role not in ("user", "assistant", "model"):
            role = "user"
        if role == "model":
            role = "assistant"
        content = m.get("content")
        if isinstance(content, list):
            # accept already-blocked content
            out.append({"role": role, "content": content})
        else:
            out.append({"role": role, "content": str(content or "")})
    return {"messages": out}


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _run_one(
    *,
    url: str,
    model: str,
    context: Optional[str],
    context_file: Optional[str],
    scraped_json: str,
    output: str,
    max_output_tokens: int,
    temperature: float,
) -> int:
    # 1) Scrape
    transcript = scrape_to_transcript(url)
    save_json(transcript, scraped_json)

    # 2) Build messages for Reproducer
    msgs = to_reproducer_messages(transcript)

    # 3) Combine additional context
    addl = None
    if context_file and os.path.exists(context_file):
        try:
            with open(context_file, "r", encoding="utf-8") as f:
                addl = f.read()
        except Exception:
            addl = None
    if context and not addl:
        addl = context

    # 4) Call reproducer
    r = Reproducer()
    resp = r.reproduce(
        msgs,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        instructions=None,
        additional_context=addl,
    )

    # 5) Extract text and parse JSON
    # LiteLLM/OpenAI/Anthropic objects differ; try best-effort text
    try:
        # Anthropic path
        from reproduce.Reproducer import Reproducer as _R  # type: ignore
        out_text = _R.extract_text(resp)
    except Exception:
        out_text = ""
    if not out_text:
        # try common attributes
        out_text = getattr(resp, "output_text", None) or getattr(resp, "content", None) or ""
        if isinstance(out_text, list):
            out_text = "".join([str(x) for x in out_text])

    result_obj: Any
    try:
        result_obj = json.loads(out_text)
    except Exception:
        # save raw text if not valid JSON
        result_obj = {"raw": out_text}
    save_json(result_obj, output)
    print(f"Saved scraped -> {scraped_json}\nSaved reproduced -> {output}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="mis-scrape",
        description="MisScrape – scrape shared model chats / threads and reproduce the conversation JSON with a target model.",
    )
    p.add_argument("url", nargs="?", help="Source URL (Gemini/ChatGPT/Claude/Grok/Reddit/GitHub issue)")
    p.add_argument("--model", help="Target model id (e.g., anthropic/claude-3-opus-20240229 or openai/gpt-4o)")
    p.add_argument("--config", help="Path to YAML config (single job or a batch under 'jobs:')")
    p.add_argument("--context", help="Additional user context (string)")
    p.add_argument("--context-file", help="Path to a file whose contents will be added as extra user context")
    p.add_argument("--scraped-json", default="scraped_transcript.json", help="Where to save the scraped transcript JSON")
    p.add_argument("--output", default="reproduced.json", help="Where to save reproduced JSON output from target model")
    p.add_argument("--max-output-tokens", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=0.0)

    args = p.parse_args(argv)

    # YAML config path: support single or batch jobs
    if args.config:
        try:
            import yaml  # type: ignore
        except Exception:
            print("Install pyyaml: %pip install pyyaml", file=sys.stderr)
            return 2
        cfg_path = os.path.abspath(args.config)
        if not os.path.exists(cfg_path):
            print(f"Config not found: {cfg_path}", file=sys.stderr)
            return 2
        with open(cfg_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}

        # Batch: jobs: [ {url, model, ...}, ... ]
        jobs = None
        if isinstance(doc, dict) and isinstance(doc.get("jobs"), list):
            jobs = doc["jobs"]
        if jobs is None:
            # Single job config at top-level
            jobs = [doc]

        exit_code = 0
        for i, job in enumerate(jobs):
            if not isinstance(job, dict):
                print(f"Skipping non-dict job at index {i}", file=sys.stderr)
                continue
            try:
                url = str(job.get("url") or job.get("source") or "").strip()
                model = str(job.get("model") or "").strip()
                if not url or not model:
                    print(f"Job {i}: missing url/model", file=sys.stderr)
                    exit_code = exit_code or 2
                    continue
                context = job.get("context")
                context_file = job.get("context_file") or job.get("context-file")
                scraped_json = job.get("scraped_json") or job.get("scraped-json") or f"scraped_{i}.json"
                output = job.get("output") or f"reproduced_{i}.json"
                max_output_tokens = int(job.get("max_output_tokens") or job.get("max-output-tokens") or args.max_output_tokens)
                temperature = float(job.get("temperature") if job.get("temperature") is not None else args.temperature)
                _run_one(
                    url=url,
                    model=model,
                    context=context,
                    context_file=context_file,
                    scraped_json=str(scraped_json),
                    output=str(output),
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
            except SystemExit as e:
                exit_code = exit_code or int(getattr(e, "code", 1) or 1)
            except Exception:
                traceback.print_exc()
                exit_code = exit_code or 1
        return exit_code

    # CLI mode
    if not args.url or not args.model:
        p.error("url and --model are required unless --config is provided")
    return _run_one(
        url=str(args.url),
        model=str(args.model),
        context=args.context,
        context_file=args.context_file,
        scraped_json=str(args.scraped_json),
        output=str(args.output),
        max_output_tokens=int(args.max_output_tokens),
        temperature=float(args.temperature),
    )


if __name__ == "__main__":
    raise SystemExit(main())


