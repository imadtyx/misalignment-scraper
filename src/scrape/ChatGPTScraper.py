from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import asyncio
from playwright.sync_api import Page, Response, sync_playwright
from playwright.async_api import async_playwright


_CHATGPT_RE = re.compile(r"^https?://(chat|chatgpt)\.openai\.com/share/", re.IGNORECASE)


def _is_share_url(url: str) -> bool:
    return bool(_CHATGPT_RE.match(url.strip()))


def _flatten_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = "\n".join([line.rstrip() for line in s.split("\n")])
    return s.strip()


@dataclass
class Transcript:
    url: str
    title: Optional[str]
    messages: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {"url": self.url, "title": self.title, "messages": self.messages}


def _extract_from_next_data(html: str) -> Optional[List[Dict[str, Any]]]:
    # Try to parse __NEXT_DATA__ if present on share pages
    m = re.search(r"<script id=\"__NEXT_DATA__\" type=\"application/json\">(.*?)</script>", html, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except Exception:
        return None
    # Heuristic search for messages arrays
    def walk(o):
        stack = [o]
        while stack:
            cur = stack.pop()
            yield cur
            if isinstance(cur, dict):
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
    candidates: List[Dict[str, Any]] = []
    for node in walk(data):
        if isinstance(node, list):
            # look for list of {author: role, content: {parts: [{text}]}} or similar
            msg_like = []
            ok = False
            for item in node:
                if not isinstance(item, dict):
                    msg_like = []
                    break
                role = item.get("role") or item.get("author") or item.get("message_author")
                content = item.get("content") or item.get("parts") or item.get("text")
                text = None
                if isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and isinstance(p.get("text"), str):
                            text = (text or "") + ("\n\n" if text else "") + p["text"]
                        elif isinstance(p, str):
                            text = (text or "") + ("\n\n" if text else "") + p
                elif isinstance(content, dict) and isinstance(content.get("text"), str):
                    text = content.get("text")
                elif isinstance(content, str):
                    text = content
                if role and text:
                    msg_like.append({"role": str(role), "content": _flatten_text(text)})
                    ok = True
            if ok and msg_like:
                candidates.append({"messages": msg_like})
    if not candidates:
        return None
    best = max(candidates, key=lambda x: len(json.dumps(x.get("messages", []), ensure_ascii=False)))
    return best.get("messages")


def _extract_from_dom(page: Page) -> List[Dict[str, Any]]:
    # Prefer ChatGPT share’s explicit author-role attribute if present
    blocks: List[Dict[str, Any]] = page.evaluate(
        """
        (() => {
          function txt(el){ return (el.innerText || el.textContent || '').trim(); }
          const out = [];
          const nodes = document.querySelectorAll('[data-message-author-role]');
          if (nodes && nodes.length){
            for (const n of nodes){
              const role = n.getAttribute('data-message-author-role') || '';
              const t = txt(n);
              if (!t) continue;
              out.push({role, text: t});
            }
            return out;
          }
          // Fallback: attempt to get role via aria-labels / headings
          const segs = [];
          const all = Array.from(document.querySelectorAll('main *'));
          for (const el of all){
            const t = txt(el);
            if (!t || t.length < 8) continue;
            const r = el.getBoundingClientRect();
            if (!r || r.height <= 0) continue;
            segs.push({role: '', text: t, top: Math.round(r.top)});
          }
          segs.sort((a,b)=>a.top-b.top);
          return segs;
        })()
        """
    )
    messages: List[Dict[str, Any]] = []
    if blocks and isinstance(blocks, list) and isinstance(blocks[0], dict) and "role" in blocks[0]:
        # If roles are provided, map them to user/model
        for b in blocks:
            role_raw = (b.get("role") or "").lower()
            text = _flatten_text(b.get("text", ""))
            if not text:
                continue
            if role_raw in ("user", "system"):  # system treat as user for continuity
                role = "user"
            elif role_raw in ("assistant", "gpt", "chatgpt", "model"):
                role = "model"
            else:
                role = "model"
            messages.append({"role": role, "content": text})
    else:
        # No roles available; alternate by order
        role = "user"
        for b in blocks:
            text = _flatten_text(b.get("text", ""))
            if not text:
                continue
            messages.append({"role": role, "content": text})
            role = "model" if role == "user" else "user"
    return messages


class ChatGPTScraper:
    def __init__(self, headless: bool = True, timeout_sec: int = 45):
        self.headless = headless
        self.timeout_sec = timeout_sec

    def scrape(self, share_url: str) -> Transcript:
        if not _is_share_url(share_url):
            raise ValueError("Provide a valid ChatGPT share URL like https://chat.openai.com/share/<id>")

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=self.headless, args=["--no-sandbox", "--disable-gpu"])  # type: ignore
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(self.timeout_sec * 1000)

            page.goto(share_url, wait_until="domcontentloaded", timeout=self.timeout_sec * 1000)
            try:
                page.wait_for_selector("main", timeout=self.timeout_sec * 1000)
            except Exception:
                pass

            title = None
            try:
                title = page.title()
            except Exception:
                title = None

            html = page.content()
            messages = _extract_from_next_data(html) or []
            if not messages:
                messages = _extract_from_dom(page)

            browser.close()

        # cleanup roles/text
        cleaned: List[Dict[str, Any]] = []
        for m in messages:
            text = _flatten_text(m.get("content") or m.get("text") or "")
            role = (m.get("role") or "model").lower()
            if text:
                cleaned.append({"role": "user" if role == "user" else "model", "content": text})

        return Transcript(url=share_url, title=title, messages=cleaned)

    async def scrape_async(self, share_url: str) -> Transcript:
        if not _is_share_url(share_url):
            raise ValueError("Provide a valid ChatGPT share URL like https://chat.openai.com/share/<id>")

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self.headless, args=["--no-sandbox", "--disable-gpu"])  # type: ignore
            context = await browser.new_context()
            page = await context.new_page()
            page.set_default_timeout(self.timeout_sec * 1000)

            await page.goto(share_url, wait_until="domcontentloaded", timeout=self.timeout_sec * 1000)
            try:
                await page.wait_for_selector("main", timeout=self.timeout_sec * 1000)
            except Exception:
                pass

            try:
                title = await page.title()
            except Exception:
                title = None

            html = await page.content()
            messages = _extract_from_next_data(html) or []
            if not messages:
                # Reuse sync dom extractor by evaluating js directly
                blocks = await page.evaluate(
                    """
                    (() => {
                      function txt(el){ return (el.innerText || el.textContent || '').trim(); }
                      const out = [];
                      const nodes = document.querySelectorAll('[data-message-author-role]');
                      if (nodes && nodes.length){
                        for (const n of nodes){
                          const role = n.getAttribute('data-message-author-role') || '';
                          const t = txt(n);
                          if (!t) continue;
                          out.push({role, text: t});
                        }
                        return out;
                      }
                      const segs = [];
                      const all = Array.from(document.querySelectorAll('main *'));
                      for (const el of all){
                        const t = txt(el);
                        if (!t || t.length < 8) continue;
                        const r = el.getBoundingClientRect();
                        if (!r || r.height <= 0) continue;
                        segs.push({role: '', text: t, top: Math.round(r.top)});
                      }
                      segs.sort((a,b)=>a.top-b.top);
                      return segs;
                    })()
                    """
                )
                messages = []
                if blocks:
                    # If roles present
                    has_roles = any((b.get("role") or "") for b in blocks)
                    if has_roles:
                        for b in blocks:
                            text = _flatten_text(b.get("text", ""))
                            if not text:
                                continue
                            role_raw = (b.get("role") or "").lower()
                            role = "user" if role_raw == "user" else "model"
                            messages.append({"role": role, "content": text})
                    else:
                        role = "user"
                        for b in blocks:
                            text = _flatten_text(b.get("text", ""))
                            if not text:
                                continue
                            messages.append({"role": role, "content": text})
                            role = "model" if role == "user" else "user"

            await browser.close()

        cleaned: List[Dict[str, Any]] = []
        for m in messages:
            text = _flatten_text(m.get("content") or m.get("text") or "")
            role = (m.get("role") or "model").lower()
            if text:
                cleaned.append({"role": "user" if role == "user" else "model", "content": text})

        return Transcript(url=share_url, title=title, messages=cleaned)


