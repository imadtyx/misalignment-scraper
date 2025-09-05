from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


_REDDIT_POST_RE = re.compile(r"https?://(www\.)?reddit\.com/r/[^/]+/comments/(?P<id>[a-z0-9]+)/?", re.IGNORECASE)


def _flatten_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join([line.rstrip() for line in s.split("\n")]).strip()


@dataclass
class RedditTranscript:
    url: str
    title: Optional[str]
    messages: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {"url": self.url, "title": self.title, "messages": self.messages}


class RedditScraper:
    """
    Scrapes a Reddit post (OP + comments) via the public .json endpoint.
    Returns a normalized transcript: { url, title, messages:[{role, author, content, metadata}, ...] }.
    """

    def __init__(self, timeout_sec: int = 30):
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "misalignment-scraper/1.0 (https://github.com/)"
        })

    def _post_json_url(self, url_or_id: str) -> str:
        m = _REDDIT_POST_RE.match(url_or_id)
        if m:
            base = url_or_id.split("?")[0].rstrip("/")
            return base + ".json"
        # If only id given, construct a canonical URL
        if re.fullmatch(r"[a-z0-9]{5,8}", url_or_id):
            return f"https://www.reddit.com/comments/{url_or_id}.json"
        raise ValueError("Provide a full Reddit post URL or a valid post id")

    def _walk_comments(self, node: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
        if not isinstance(node, dict):
            return
        data = node.get("data") or {}
        kind = node.get("kind")
        if kind == "t1":  # comment
            author = data.get("author")
            body = _flatten_text(data.get("body"))
            created_utc = data.get("created_utc")
            if body:
                out.append({
                    "role": "commenter",
                    "author": author,
                    "content": body,
                    "metadata": {
                        "score": data.get("score"),
                        "created_utc": created_utc,
                        "permalink": data.get("permalink"),
                    },
                })
            # replies
            replies = data.get("replies")
            if isinstance(replies, dict):
                children = ((replies.get("data") or {}).get("children") or [])
                for ch in children:
                    self._walk_comments(ch, out)
        elif kind in ("Listing", None):
            children = ((data.get("data") or {}).get("children") or []) if kind == "Listing" else (node.get("children") or [])
            for ch in children:
                self._walk_comments(ch, out)
        elif kind == "more":
            # Skipping "more" placeholders; full expansion would need additional API calls
            return

    def scrape(self, url_or_id: str) -> RedditTranscript:
        jurl = self._post_json_url(url_or_id)
        r = self.session.get(jurl, timeout=self.timeout_sec)
        r.raise_for_status()
        j = r.json()
        if not isinstance(j, list) or len(j) < 2:
            raise RuntimeError("Unexpected Reddit JSON structure")

        post_listing = j[0]
        comments_listing = j[1]

        post_children = ((post_listing.get("data") or {}).get("children") or [])
        if not post_children:
            raise RuntimeError("No post found")
        post = (post_children[0].get("data") or {})

        title = post.get("title")
        selftext = _flatten_text(post.get("selftext"))
        op_author = post.get("author")
        permalink = post.get("permalink")
        canonical_url = f"https://www.reddit.com{permalink}" if permalink else url_or_id

        messages: List[Dict[str, Any]] = []
        if selftext:
            messages.append({
                "role": "op",
                "author": op_author,
                "content": selftext,
                "metadata": {
                    "score": post.get("score"),
                    "created_utc": post.get("created_utc"),
                    "permalink": permalink,
                },
            })

        # Flatten comments in display order
        comments_out: List[Dict[str, Any]] = []
        for ch in ((comments_listing.get("data") or {}).get("children") or []):
            self._walk_comments(ch, comments_out)
        messages.extend(comments_out)

        return RedditTranscript(url=canonical_url, title=title, messages=messages)

    @staticmethod
    def save_transcript(t: RedditTranscript, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(t.to_dict(), f, ensure_ascii=False, indent=2)


