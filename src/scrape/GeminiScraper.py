from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from playwright.sync_api import sync_playwright, Page, Response
from playwright.async_api import async_playwright
import asyncio


_URL_RE = re.compile(r"^https?://gemini\.google\.com/share/[a-z0-9]+$", re.IGNORECASE)


def _is_share_url(url: str) -> bool:
    return bool(_URL_RE.match(url.strip()))


def _flatten_text(s: str) -> str:
    if s is None:
        return ""
    # Normalize whitespace, collapse multiple newlines, and strip
    s = re.sub(r"\r\n|\r", "\n", s)
    s = re.sub(r"\u00a0", " ", s)
    # Collapse 3+ newlines to 2
    s = re.sub(r"\n{3,}", "\n\n", s)
    # Trim trailing spaces on lines
    s = "\n".join([line.rstrip() for line in s.split("\n")])
    return s.strip()


def _walk(obj: Any):
    stack = [obj]
    while stack:
        cur = stack.pop()
        yield cur
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def _extract_messages_from_network_payload(payloads: List[Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Best-effort extraction: scan captured JSON payloads for structures that look like
    Gemini messages. We look for objects that contain a list of message-like entries
    with role/user distinction and text parts.
    """
    candidates: List[Dict[str, Any]] = []

    for p in payloads:
        try:
            for node in _walk(p):
                if not isinstance(node, dict):
                    continue
                # Heuristic 1: {messages: [{role, parts|content|text, ...}]}
                if "messages" in node and isinstance(node["messages"], list):
                    msgs = []
                    ok = False
                    for m in node["messages"]:
                        if not isinstance(m, dict):
                            continue
                        role = m.get("role") or m.get("author") or m.get("speaker")
                        parts = m.get("parts") or m.get("content") or m.get("text")
                        text = None
                        if isinstance(parts, list):
                            # parts may be like [{text: ...}, {inline_data: ...}]
                            for part in parts:
                                if isinstance(part, dict) and isinstance(part.get("text"), str):
                                    text = (text or "") + ("\n\n" if text else "") + part["text"]
                        elif isinstance(parts, dict) and isinstance(parts.get("text"), str):
                            text = parts.get("text")
                        elif isinstance(parts, str):
                            text = parts

                        if role and text:
                            msgs.append({
                                "role": str(role),
                                "content": _flatten_text(text),
                            })
                            ok = True
                    if ok and msgs:
                        candidates.append({"messages": msgs})

                # Heuristic 2: look for {candidates: [{content: {parts: [{text}]}}]} (PaLM-like)
                if "candidates" in node and isinstance(node["candidates"], list):
                    msgs = []
                    ok = False
                    for c in node["candidates"]:
                        if not isinstance(c, dict):
                            continue
                        role = c.get("author") or c.get("role") or "model"
                        content = c.get("content") or {}
                        if isinstance(content, dict) and isinstance(content.get("parts"), list):
                            # concat all text parts
                            texts = [
                                part.get("text")
                                for part in content["parts"]
                                if isinstance(part, dict) and isinstance(part.get("text"), str)
                            ]
                            text = _flatten_text("\n\n".join([t for t in texts if t]))
                            if text:
                                msgs.append({"role": str(role), "content": text})
                                ok = True
                    if ok and msgs:
                        candidates.append({"messages": msgs})
        except Exception:
            continue

    # Pick the longest transcript among candidates
    if candidates:
        best = max(candidates, key=lambda x: len(json.dumps(x.get("messages", []), ensure_ascii=False)))
        return best.get("messages")
    return None


def _extract_messages_from_dom(page: Page) -> List[Dict[str, Any]]:
    """
    Fallback DOM extraction: best-effort traversal that pierces shadow roots,
    collects visible message blocks as alternating user/model messages.
    This is heuristic and may need adjustment if Gemini updates markup.
    """
    script = """
    (() => {
      function allText(el) {
        return (el.innerText || el.textContent || '').trim();
      }
      function walkShadow(node, acc) {
        if (!node) return;
        acc.push(node);
        const sr = node.shadowRoot;
        if (sr) {
          for (const ch of sr.children) walkShadow(ch, acc);
        }
        for (const ch of node.children || []) walkShadow(ch, acc);
      }
      const roots = [];
      walkShadow(document.querySelector('chat-app') || document.body, roots);

      // Collect blocks that look like message bubbles: non-empty text, not nav/footer
      const items = [];
      for (const n of roots) {
        const t = allText(n);
        if (!t) continue;
        // Skip tiny or chrome text
        if (t.length < 10) continue;
        // Exclude global headers/menus by keywords
        const lower = t.toLowerCase();
        if (lower.includes('sign in') || lower.includes('google') && lower.includes('privacy')) continue;
        // Likely message blocks have multiple lines or punctuation
        const lines = t.split(/\n+/).filter(Boolean);
        if (lines.length < 1) continue;
        items.push({t, linesCount: lines.length, charCount: t.length});
      }
      // Sort by DOM order approximated by discovery sequence and keep unique texts
      const seen = new Set();
      const uniq = [];
      for (const it of items) {
        if (seen.has(it.t)) continue; // avoid duplicates from nested containers
        seen.add(it.t);
        uniq.push(it.t);
      }
      // Heuristic: messages usually appear sequentially and are separated visually.
      // Return top N longest blocks as the transcript backbone.
      uniq.sort((a, b) => b.length - a.length);
      const top = uniq.slice(0, 40).sort((a,b) => a.length - b.length);
      return top;
    })()
    """
    blocks: List[str] = page.evaluate(script)
    # Map to alternating roles: user, model, user, model...
    messages: List[Dict[str, Any]] = []
    role = "user"
    for b in blocks:
        text = _flatten_text(b)
        if not text:
            continue
        messages.append({"role": role, "content": text})
        role = "model" if role == "user" else "user"
    return messages


@dataclass
class GeminiTranscript:
    url: str
    title: Optional[str]
    messages: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "messages": self.messages,
        }


class GeminiScraper:
    """
    Scrape a public Gemini shared chat URL into a clean JSON transcript.
    Uses Playwright to load the page, capture network JSON payloads, and fallback to DOM parsing.
    """

    def __init__(self, headless: bool = True, timeout_sec: int = 30):
        self.headless = headless
        self.timeout_sec = timeout_sec

    def _collect_json_responses(self, page: Page) -> List[Any]:
        payloads: List[Any] = []

        def on_response(resp: Response):
            try:
                ct = resp.headers.get("content-type", "")
                if "application/json" in ct:
                    # Only parse small-ish bodies (avoid streaming)
                    txt = resp.text()
                    if not txt:
                        return
                    data = json.loads(txt)
                    payloads.append(data)
            except Exception:
                pass

        page.on("response", on_response)
        return payloads

    def scrape(self, share_url: str) -> GeminiTranscript:
        if not _is_share_url(share_url):
            raise ValueError("Provide a valid Gemini share URL like https://gemini.google.com/share/<id>")

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=self.headless, args=["--no-sandbox", "--disable-gpu"])  # safer under root/CI
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(self.timeout_sec * 1000)

            payloads = self._collect_json_responses(page)

            # Navigate and wait for network to be mostly idle
            page.goto(share_url, wait_until="domcontentloaded", timeout=self.timeout_sec * 1000)
            # Wait for the app shell instead of networkidle (SPAs may never be idle)
            try:
                page.wait_for_selector("chat-app", timeout=self.timeout_sec * 1000)
            except Exception:
                pass
            # Short settle time to allow initial fetches
            time.sleep(2)

            # Heuristic scroll to force lazy content
            try:
                for _ in range(3):
                    page.mouse.wheel(0, 1500)
                    time.sleep(0.25)
            except Exception:
                pass

            # Extract title
            title = None
            try:
                title = page.title()
            except Exception:
                title = None

            # Try network-derived messages first
            messages = _extract_messages_from_network_payload(payloads) or []
            if not messages:
                # Fallback to DOM extraction (bubbles), else permissive text scrape
                try:
                    messages = _extract_messages_from_dom(page)
                except Exception:
                    messages = []
                if not messages:
                    try:
                        segments = page.evaluate(
                            """
                            (() => {
                              function textOf(el){ return (el.innerText || el.textContent || '').trim(); }
                              const host = document.querySelector('chat-app') || document.body;
                              const nodes = [];
                              function walkAll(node){
                                nodes.push(node);
                                if (node.shadowRoot){ for (const ch of node.shadowRoot.children) walkAll(ch); }
                                for (const ch of (node.children||[])) walkAll(ch);
                              }
                              walkAll(host);
                              const segs = [];
                              for (const n of nodes){
                                const t = textOf(n);
                                if (!t || t.length < 12) continue;
                                const lower = t.toLowerCase();
                                if (lower.includes('sign in')) continue;
                                const r = n.getBoundingClientRect();
                                if (!r || r.height <= 0) continue;
                                segs.push({ t, top: Math.round(r.top), height: Math.round(r.height) });
                              }
                              // sort top-to-bottom and de-dupe by exact text
                              segs.sort((a,b)=>a.top-b.top);
                              const seen = new Set();
                              const out = [];
                              for (const s of segs){ if (!seen.has(s.t)) { seen.add(s.t); out.push(s); } }
                              // limit to 200 segments to avoid noise
                              return out.slice(0,200);
                            })()
                            """
                        )
                        if segments:
                            # group by vertical gaps into message clusters
                            clusters: List[List[Dict[str, Any]]] = []
                            cur: List[Dict[str, Any]] = []
                            last_top: Optional[int] = None
                            for s in segments:
                                top = int(s.get("top", 0))
                                if last_top is None:
                                    cur.append(s)
                                    last_top = top
                                    continue
                                gap = top - last_top
                                if gap > 40 and cur:
                                    clusters.append(cur)
                                    cur = [s]
                                else:
                                    cur.append(s)
                                last_top = top
                            if cur:
                                clusters.append(cur)
                            role = "user"
                            tmp = []
                            for cl in clusters:
                                text = _flatten_text("\n".join([c.get("t","") for c in cl]))
                                if text:
                                    tmp.append({"role": role, "content": text})
                                    role = "model" if role == "user" else "user"
                            messages = tmp
                    except Exception:
                        messages = []

            browser.close()

        # Final cleanup: ensure content strings are non-empty and trim
        cleaned: List[Dict[str, Any]] = []
        for m in messages:
            text = _flatten_text(m.get("content", ""))
            role = (m.get("role") or "model").lower()
            if text:
                cleaned.append({"role": role, "content": text})

        # If everything failed, at least return an empty set with URL
        return GeminiTranscript(url=share_url, title=title, messages=cleaned)

    @staticmethod
    def save_transcript(transcript: GeminiTranscript, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=2)

    # --- Async variant for notebooks / running inside an event loop ---
    async def scrape_async(self, share_url: str) -> GeminiTranscript:
        if not _is_share_url(share_url):
            raise ValueError("Provide a valid Gemini share URL like https://gemini.google.com/share/<id>")

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self.headless, args=["--no-sandbox", "--disable-gpu"])  # type: ignore
            context = await browser.new_context()
            page = await context.new_page()
            page.set_default_timeout(self.timeout_sec * 1000)

            payloads: List[Any] = []

            async def on_response(resp):
                try:
                    ct = (await resp.all_headers()).get("content-type", "")
                    if "application/json" in ct:
                        txt = await resp.text()
                        if not txt:
                            return
                        try:
                            data = json.loads(txt)
                            payloads.append(data)
                        except Exception:
                            pass
                except Exception:
                    pass

            page.on("response", on_response)

            await page.goto(share_url, wait_until="domcontentloaded", timeout=self.timeout_sec * 1000)
            try:
                await page.wait_for_selector("chat-app", timeout=self.timeout_sec * 1000)
            except Exception:
                pass
            await asyncio.sleep(2)

            try:
                for _ in range(3):
                    await page.mouse.wheel(0, 1500)
                    await asyncio.sleep(0.25)
            except Exception:
                pass

            try:
                title = await page.title()
            except Exception:
                title = None

            messages = _extract_messages_from_network_payload(payloads) or []
            if not messages:
                try:
                    messages = await page.evaluate(
                        """
                        (() => {
                          function allText(el) { return (el.innerText || el.textContent || '').trim(); }
                          function walkShadow(node, acc) {
                            if (!node) return;
                            acc.push(node);
                            const sr = node.shadowRoot;
                            if (sr) {
                              for (const ch of sr.children) walkShadow(ch, acc);
                            }
                            for (const ch of node.children || []) walkShadow(ch, acc);
                          }
                          const roots = [];
                          walkShadow(document.querySelector('chat-app') || document.body, roots);

                          const items = [];
                          for (const n of roots) {
                            const t = allText(n);
                            if (!t) continue;
                            if (t.length < 10) continue;
                            const lower = t.toLowerCase();
                            if (lower.includes('sign in') || (lower.includes('google') && lower.includes('privacy'))) continue;
                            const lines = t.split(/\n+/).filter(Boolean);
                            if (!lines.length) continue;
                            items.push(t);
                          }
                          const seen = new Set();
                          const uniq = [];
                          for (const t of items) { if (!seen.has(t)) { seen.add(t); uniq.push(t); } }
                          uniq.sort((a, b) => b.length - a.length);
                          const top = uniq.slice(0, 40).sort((a, b) => a.length - b.length);
                          return top;
                        })()
                        """
                    )
                except Exception:
                    messages = []
                if messages:
                    # Map to alternating roles
                    role = "user"
                    mapped: List[Dict[str, Any]] = []
                    for b in messages:
                        text = _flatten_text(b)
                        if not text:
                            continue
                        mapped.append({"role": role, "content": text})
                        role = "model" if role == "user" else "user"
                    messages = mapped
                else:
                    # Permissive text scrape if bubble parse failed
                    try:
                        segments = await page.evaluate(
                            """
                            (() => {
                              function textOf(el){ return (el.innerText || el.textContent || '').trim(); }
                              const host = document.querySelector('chat-app') || document.body;
                              const nodes = [];
                              function walkAll(node){
                                nodes.push(node);
                                if (node.shadowRoot){ for (const ch of node.shadowRoot.children) walkAll(ch); }
                                for (const ch of (node.children||[])) walkAll(ch);
                              }
                              walkAll(host);
                              const segs = [];
                              for (const n of nodes){
                                const t = textOf(n);
                                if (!t || t.length < 12) continue;
                                const lower = t.toLowerCase();
                                if (lower.includes('sign in')) continue;
                                const r = n.getBoundingClientRect();
                                if (!r || r.height <= 0) continue;
                                segs.push({ t, top: Math.round(r.top), height: Math.round(r.height) });
                              }
                              segs.sort((a,b)=>a.top-b.top);
                              const seen = new Set();
                              const out = [];
                              for (const s of segs){ if (!seen.has(s.t)) { seen.add(s.t); out.push(s); } }
                              return out.slice(0,200);
                            })()
                            """
                        )
                        if segments:
                            clusters: List[List[Dict[str, Any]]] = []
                            cur: List[Dict[str, Any]] = []
                            last_top: Optional[int] = None
                            for s in segments:
                                top = int(s.get("top", 0))
                                if last_top is None:
                                    cur.append(s)
                                    last_top = top
                                    continue
                                gap = top - last_top
                                if gap > 40 and cur:
                                    clusters.append(cur)
                                    cur = [s]
                                else:
                                    cur.append(s)
                                last_top = top
                            if cur:
                                clusters.append(cur)
                            role = "user"
                            tmp = []
                            for cl in clusters:
                                text = _flatten_text("\n".join([c.get("t","") for c in cl]))
                                if text:
                                    tmp.append({"role": role, "content": text})
                                    role = "model" if role == "user" else "user"
                            messages = tmp
                    except Exception:
                        messages = []

            await browser.close()

        cleaned: List[Dict[str, Any]] = []
        for m in messages:
            text = _flatten_text(m.get("content", ""))
            role = (m.get("role") or "model").lower()
            if text:
                cleaned.append({"role": role, "content": text})

        return GeminiTranscript(url=share_url, title=title, messages=cleaned)


