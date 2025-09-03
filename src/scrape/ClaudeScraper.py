from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import asyncio
from playwright.sync_api import Page, sync_playwright
from playwright.async_api import async_playwright


_CLAUDE_RE = re.compile(r"^https?://claude\.ai/(share|chat)/", re.IGNORECASE)


def _is_share_url(url: str) -> bool:
    return bool(_CLAUDE_RE.match(url.strip()))


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


def _extract_from_dom(page: Page) -> List[Dict[str, Any]]:
    # Claude share pages usually render bubbles; capture visible blocks with role hints when present
    blocks: List[Dict[str, Any]] = page.evaluate(
        """
        (() => {
          function txt(el){ return (el.innerText || el.textContent || '').trim(); }
          const segs = [];
          const root = document.querySelector('main') || document.body;
          const all = Array.from(root.querySelectorAll('*'));
          for (const el of all){
            const t = txt(el);
            if (!t || t.length < 8) continue;
            const r = el.getBoundingClientRect();
            if (!r || r.height <= 0) continue;
            const roleAttr = el.getAttribute('data-author') || el.getAttribute('aria-label') || '';
            segs.push({text: t, roleAttr, top: Math.round(r.top)});
          }
          segs.sort((a,b)=>a.top-b.top);
          return segs;
        })()
        """
    )
    # Cluster by gaps and assign roles heuristically
    clusters: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    last_top: Optional[int] = None
    for s in blocks:
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

    messages: List[Dict[str, Any]] = []
    role = "user"
    for cl in clusters:
        text = _flatten_text("\n".join([c.get("text", "") for c in cl]))
        if not text:
            continue
        # If any segment in cluster hints at assistant, use model
        rattr = " ".join([(c.get("roleAttr") or "").lower() for c in cl])
        if any(k in rattr for k in ("assistant", "model", "claude")):
            role = "model"
        elif any(k in rattr for k in ("user", "human")):
            role = "user"
        messages.append({"role": role, "content": text})
        role = "model" if role == "user" else "user"
    return messages


class ClaudeScraper:
    def __init__(self, headless: bool = True, timeout_sec: int = 45):
        self.headless = headless
        self.timeout_sec = timeout_sec

    def scrape(self, share_url: str) -> Transcript:
        if not _is_share_url(share_url):
            raise ValueError("Provide a valid Claude share URL like https://claude.ai/share/<id>")
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

            try:
                title = page.title()
            except Exception:
                title = None

            messages = _extract_from_dom(page)
            browser.close()
        cleaned: List[Dict[str, Any]] = []
        for m in messages:
            text = _flatten_text(m.get("content") or m.get("text") or "")
            role = (m.get("role") or "model").lower()
            if text:
                cleaned.append({"role": "user" if role == "user" else "model", "content": text})
        return Transcript(url=share_url, title=title, messages=cleaned)

    async def scrape_async(self, share_url: str) -> Transcript:
        if not _is_share_url(share_url):
            raise ValueError("Provide a valid Claude share URL like https://claude.ai/share/<id>")
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

            blocks = await page.evaluate(
                """
                (() => {
                  function txt(el){ return (el.innerText || el.textContent || '').trim(); }
                  const segs = [];
                  const root = document.querySelector('main') || document.body;
                  const all = Array.from(root.querySelectorAll('*'));
                  for (const el of all){
                    const t = txt(el);
                    if (!t || t.length < 8) continue;
                    const r = el.getBoundingClientRect();
                    if (!r || r.height <= 0) continue;
                    const roleAttr = el.getAttribute('data-author') || el.getAttribute('aria-label') || '';
                    segs.push({text: t, roleAttr, top: Math.round(r.top)});
                  }
                  segs.sort((a,b)=>a.top-b.top);
                  return segs;
                })()
                """
            )
            # Cluster and assign roles
            clusters: List[List[Dict[str, Any]]] = []
            cur: List[Dict[str, Any]] = []
            last_top: Optional[int] = None
            for s in blocks:
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

            messages: List[Dict[str, Any]] = []
            role = "user"
            for cl in clusters:
                text = _flatten_text("\n".join([c.get("text", "") for c in cl]))
                if not text:
                    continue
                rattr = " ".join([(c.get("roleAttr") or "").lower() for c in cl])
                if any(k in rattr for k in ("assistant", "model", "claude")):
                    role = "model"
                elif any(k in rattr for k in ("user", "human")):
                    role = "user"
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


