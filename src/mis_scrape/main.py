from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# Scrapers
try:
    from src.scrape.scrapers.GeminiScraper import GeminiScraper  # type: ignore
except Exception:
    from src.scrape.scrapers.GeminiScraper import GeminiScraper  # noqa: F401
try:
    from src.scrape.scrapers.ChatGPTScraper import ChatGPTScraper  # type: ignore
    from src.scrape.scrapers.ClaudeScraper import ClaudeScraper  # type: ignore
    from src.scrape.scrapers.GrokScraper import GrokScraper  # type: ignore
except Exception:
    pass
try:
    from src.scrape.RedditScraper import RedditScraper  # type: ignore
except Exception:
    pass
try:
    from src.scrape.GithubScraper import GithubScraper  # type: ignore
except Exception:
    pass

from src.reproduce.Reproducer import Reproducer  # type: ignore


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


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="mis-scrape",
        description="MisScrape – scrape shared model chats / threads and reproduce the conversation JSON with a target model.",
    )
    p.add_argument("url", help="Source URL (Gemini/ChatGPT/Claude/Grok/Reddit/GitHub issue)")
    p.add_argument("--model", required=True, help="Target model id (e.g., anthropic/claude-3-opus-20240229 or openai/gpt-4o)")
    p.add_argument("--context", help="Additional user context (string)")
    p.add_argument("--context-file", help="Path to a file whose contents will be added as extra user context")
    p.add_argument("--scraped-json", default="scraped_transcript.json", help="Where to save the scraped transcript JSON")
    p.add_argument("--output", default="reproduced.json", help="Where to save reproduced JSON output from target model")
    p.add_argument("--max-output-tokens", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=0.0)

    args = p.parse_args(argv)

    # 1) Scrape
    transcript = scrape_to_transcript(args.url)
    save_json(transcript, args.scraped_json)

    # 2) Build messages for Reproducer
    msgs = to_reproducer_messages(transcript)

    # 3) Combine additional context
    addl = None
    if args.context_file and os.path.exists(args.context_file):
        try:
            with open(args.context_file, "r", encoding="utf-8") as f:
                addl = f.read()
        except Exception:
            addl = None
    if args.context and not addl:
        addl = args.context

    # 4) Call reproducer
    r = Reproducer()
    resp = r.reproduce(
        msgs,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        instructions=None,
        additional_context=addl,
    )

    # 5) Extract text and parse JSON
    # LiteLLM/OpenAI/Anthropic objects differ; try best-effort text
    try:
        # Anthropic path
        from src.reproduce.Reproducer import Reproducer as _R  # type: ignore
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
    save_json(result_obj, args.output)
    print(f"Saved scraped -> {args.scraped_json}\nSaved reproduced -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


