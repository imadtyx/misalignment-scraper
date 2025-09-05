from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


_ISSUE_RE = re.compile(r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/issues/(?P<number>\d+)")


def _flatten_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join([line.rstrip() for line in s.split("\n")]).strip()


@dataclass
class GithubTranscript:
    url: str
    title: Optional[str]
    messages: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {"url": self.url, "title": self.title, "messages": self.messages}


class GithubScraper:
    """
    Scrapes a GitHub issue with comments using GitHub REST API v3.
    If GITHUB_TOKEN is set in the environment, it will be used to raise rate limits.
    Output is normalized to { url, title, messages:[{role, author, content, metadata}, ...] }.
    """

    def __init__(self, timeout_sec: int = 30, github_token: Optional[str] = None):
        self.timeout_sec = timeout_sec
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "User-Agent": "misalignment-scraper/1.0"
        })
        if github_token:
            self.session.headers["Authorization"] = f"Bearer {github_token}"

    def _parse_issue_url(self, url: str) -> Dict[str, Any]:
        m = _ISSUE_RE.match(url)
        if not m:
            raise ValueError("Provide a valid GitHub issue URL")
        return m.groupdict()

    def scrape(self, issue_url: str) -> GithubTranscript:
        parts = self._parse_issue_url(issue_url)
        owner, repo, number = parts["owner"], parts["repo"], parts["number"]

        issue_api = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
        comments_api = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments"

        ir = self.session.get(issue_api, timeout=self.timeout_sec)
        ir.raise_for_status()
        issue = ir.json()

        cr = self.session.get(comments_api, timeout=self.timeout_sec)
        cr.raise_for_status()
        comments = cr.json()

        title = issue.get("title")
        messages: List[Dict[str, Any]] = []

        body = _flatten_text(issue.get("body"))
        if body:
            messages.append({
                "role": "op",
                "author": (issue.get("user") or {}).get("login"),
                "content": body,
                "metadata": {
                    "created_at": issue.get("created_at"),
                    "state": issue.get("state"),
                    "labels": [l.get("name") for l in (issue.get("labels") or []) if isinstance(l, dict)],
                },
            })

        for c in comments or []:
            cbody = _flatten_text(c.get("body"))
            if not cbody:
                continue
            messages.append({
                "role": "commenter",
                "author": (c.get("user") or {}).get("login"),
                "content": cbody,
                "metadata": {
                    "created_at": c.get("created_at"),
                    "reactions": (c.get("reactions") or {}).get("total_count"),
                },
            })

        return GithubTranscript(url=issue_url, title=title, messages=messages)

    @staticmethod
    def save_transcript(t: GithubTranscript, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(t.to_dict(), f, ensure_ascii=False, indent=2)


