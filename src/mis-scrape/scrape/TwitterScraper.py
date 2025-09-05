from __future__ import annotations

import os
import re
import json
import base64
from typing import Any, Dict, List, Optional

import requests


# --- helpers (copied from notebook) -----------------------------------------
_URL_ID_RE = re.compile(r"(?:x|twitter)\.com/.+?/status/(\d+)")


def _tid(s: str) -> str:
    s = s.strip()
    if s.isdigit():
        return s
    m = _URL_ID_RE.search(s)
    if not m:
        raise ValueError("Provide a Tweet ID or a valid Tweet URL")
    return m.group(1)


def _tolist(objs):
    return [o.data if hasattr(o, "data") else dict(o) for o in (objs or [])]


def _chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


class TwitterScraper:
    """A wrapper class containing copies of the notebook functions up to
    and including tweets_to_anthropic_messages, so they can be used from code.

    These are copied (not imported) to avoid modifying the notebook and to
    allow straightforward reuse.
    """

    # --- main fetch (copied) ------------------------------------------------
    @staticmethod
    def fetch_all_tweet_data_basic(
        url_or_id: str,
        include_thread: bool = True,
        hydrate_referenced: bool = True,
        fetch_images: bool = False,
        max_images: int = 25,
    ) -> Dict[str, Any]:
        """Fetches EVERYTHING available on v2 with a Basic app-only bearer.
        No files. No OCR. No LLM formatting.
        """
        try:
            import tweepy  # type: ignore
        except Exception as e:  # pragma: no cover - depends on env
            raise RuntimeError("Install tweepy: %pip install tweepy") from e

        token = os.getenv("X_BEARER")
        if not token:
            raise RuntimeError("Set X_BEARER env var to your X API Bearer token (decoded, not %3D)")
        client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)

        tid = _tid(url_or_id)

        tweet_fields = [
            "attachments",
            "author_id",
            "context_annotations",
            "conversation_id",
            "created_at",
            "edit_history_tweet_ids",
            "entities",
            "geo",
            "id",
            "in_reply_to_user_id",
            "lang",
            "possibly_sensitive",
            "public_metrics",
            "referenced_tweets",
            "reply_settings",
            "source",
            "text",
            "withheld",
        ]
        user_fields = [
            "id",
            "name",
            "username",
            "verified",
            "verified_type",
            "created_at",
            "description",
            "entities",
            "location",
            "pinned_tweet_id",
            "profile_image_url",
            "protected",
            "public_metrics",
            "url",
            "withheld",
        ]
        media_fields = [
            "media_key",
            "type",
            "url",
            "preview_image_url",
            "alt_text",
            "variants",
            "width",
            "height",
            "duration_ms",
            "public_metrics",
        ]
        poll_fields = ["duration_minutes", "end_datetime", "id", "options", "voting_status"]
        place_fields = [
            "id",
            "full_name",
            "name",
            "country",
            "country_code",
            "place_type",
            "geo",
            "contained_within",
        ]
        expansions = [
            "author_id",
            "in_reply_to_user_id",
            "attachments.media_keys",
            "attachments.poll_ids",
            "entities.mentions.username",
            "geo.place_id",
            "referenced_tweets.id",
            "referenced_tweets.id.author_id",
        ]
        fields = dict(
            tweet_fields=tweet_fields,
            user_fields=user_fields,
            media_fields=media_fields,
            poll_fields=poll_fields,
            place_fields=place_fields,
            expansions=expansions,
        )

        # Root tweet
        root_resp = client.get_tweet(tid, **fields)
        if not root_resp.data:
            raise RuntimeError("Tweet not found or not accessible with this token")

        root = dict(root_resp.data.data)
        inc_users = _tolist(getattr(root_resp, "includes", {}).get("users"))
        inc_media = _tolist(getattr(root_resp, "includes", {}).get("media"))
        inc_tweets = _tolist(getattr(root_resp, "includes", {}).get("tweets"))
        inc_polls = _tolist(getattr(root_resp, "includes", {}).get("polls"))
        inc_places = _tolist(getattr(root_resp, "includes", {}).get("places"))

        conversation_id = root.get("conversation_id") or root.get("id")

        # Thread via Recent Search
        thread: List[Dict[str, Any]] = []
        pages_meta: List[Dict[str, Any]] = []
        if include_thread:
            paginator = tweepy.Paginator(
                client.search_recent_tweets,
                query=f"conversation_id:{conversation_id} -is:retweet",
                max_results=100,
                **fields,
            )
            for page in paginator:
                if page.data:
                    thread.extend([dict(t.data) for t in page.data])
                if getattr(page, "includes", None):
                    inc_users += _tolist(page.includes.get("users"))
                    inc_media += _tolist(page.includes.get("media"))
                    inc_tweets += _tolist(page.includes.get("tweets"))
                    inc_polls += _tolist(page.includes.get("polls"))
                    inc_places += _tolist(page.includes.get("places"))
                pages_meta.append(dict(getattr(page, "meta", {}) or {}))

        # De-dup includes
        users_by_id = {u["id"]: u for u in inc_users}
        media_by_key = {m["media_key"]: m for m in inc_media if "media_key" in m}
        tweets_by_id = {t["id"]: t for t in inc_tweets}
        polls_by_id = {p["id"]: p for p in inc_polls}
        places_by_id = {p["id"]: p for p in inc_places}

        # Second pass: hydrate referenced/quoted tweets fully (to capture their media/polls/etc.)
        ref_ids = set()

        def _collect_refs(t: Dict[str, Any]):
            for r in (t.get("referenced_tweets") or []):
                if r.get("id"):
                    ref_ids.add(r["id"])

        _collect_refs(root)
        for t in thread:
            _collect_refs(t)

        missing = [rid for rid in ref_ids if rid not in tweets_by_id and rid != root.get("id")]
        hydrated_refs: List[Dict[str, Any]] = []
        if hydrate_referenced and missing:
            for batch in _chunk(missing, 100):
                resp = client.get_tweets(batch, **fields)
                if resp.data:
                    got = [dict(t.data) for t in resp.data]
                    hydrated_refs.extend(got)
                    for t in got:
                        tweets_by_id.setdefault(t["id"], t)
                if getattr(resp, "includes", None):
                    for u in _tolist(resp.includes.get("users")):
                        users_by_id.setdefault(u["id"], u)
                    for m in _tolist(resp.includes.get("media")):
                        media_by_key.setdefault(m["media_key"], m)
                    for tw in _tolist(resp.includes.get("tweets")):
                        tweets_by_id.setdefault(tw["id"], tw)
                    for p in _tolist(resp.includes.get("polls")):
                        polls_by_id.setdefault(p["id"], p)
                    for pl in _tolist(resp.includes.get("places")):
                        places_by_id.setdefault(pl["id"], pl)

        # Media inventory (photos + video/gif w/ preview + variants)
        photos: List[Dict[str, Any]] = []
        videos: List[Dict[str, Any]] = []

        def _collect_media_from_tweet(t: Dict[str, Any]):
            att = (t.get("attachments") or {})
            for mk in att.get("media_keys", []) or []:
                m = media_by_key.get(mk)
                if not m:
                    continue
                if m.get("type") == "photo" and m.get("url"):
                    photos.append(
                        {
                            "tweet_id": t.get("id"),
                            "media_key": mk,
                            "url": m.get("url"),
                            "alt_text": m.get("alt_text"),
                            "width": m.get("width"),
                            "height": m.get("height"),
                        }
                    )
                elif m.get("type") in ("video", "animated_gif"):
                    videos.append(
                        {
                            "tweet_id": t.get("id"),
                            "media_key": mk,
                            "preview_image_url": m.get("preview_image_url"),
                            "variants": m.get("variants"),
                            "width": m.get("width"),
                            "height": m.get("height"),
                            "public_metrics": m.get("public_metrics"),
                        }
                    )

        _collect_media_from_tweet(root)
        for t in thread:
            _collect_media_from_tweet(t)
        for t in hydrated_refs:
            _collect_media_from_tweet(t)

        # Optional: fetch images/preview thumbs into base64 (for multimodal LLMs)
        images_b64: List[Dict[str, Any]] = []
        if fetch_images:
            queue = list(photos) + [
                {
                    "url": v.get("preview_image_url"),
                    "media_key": v.get("media_key"),
                    "tweet_id": v.get("tweet_id"),
                }
                for v in videos
                if v.get("preview_image_url")
            ]
            for item in queue[: max_images]:
                url = item.get("url")
                if not url:
                    continue
                try:
                    rb = requests.get(url, timeout=30)
                    rb.raise_for_status()
                    mime = rb.headers.get("content-type", "image/jpeg").split(";")[0]
                    images_b64.append(
                        {
                            "tweet_id": item.get("tweet_id"),
                            "media_key": item.get("media_key"),
                            "mime": mime,
                            "size_bytes": len(rb.content),
                            "base64": base64.b64encode(rb.content).decode("ascii"),
                        }
                    )
                except Exception as e:  # pragma: no cover - network
                    images_b64.append(
                        {
                            "tweet_id": item.get("tweet_id"),
                            "media_key": item.get("media_key"),
                            "error": str(e),
                        }
                    )

        return {
            "input": {"url_or_id": url_or_id, "tweet_id": tid},
            "meta": {
                "conversation_id": conversation_id,
                "root_author_id": root.get("author_id"),
                "root_created_at": root.get("created_at"),
                "root_url": f"https://x.com/i/web/status/{tid}",
            },
            "root": root,
            "thread": [t for t in thread if t.get("id") != root.get("id")],
            "includes": {
                "users": users_by_id,
                "media": media_by_key,
                "tweets": tweets_by_id,
                "polls": polls_by_id,
                "places": places_by_id,
            },
            "referenced_hydrated": hydrated_refs,
            "media_urls": {"photos": photos, "videos": videos},
            "images_b64": images_b64,
            "pages_meta": pages_meta,
            "counts": {
                "num_thread_tweets": len(thread),
                "num_users": len(users_by_id),
                "num_media": len(media_by_key),
                "num_ref_tweets": len(tweets_by_id),
                "num_polls": len(polls_by_id),
                "num_places": len(places_by_id),
                "num_photo_urls": len(photos),
                "num_video_items": len(videos),
                "num_images_b64": len(images_b64),
            },
            "notes": (
                "This is the maximum exposed by /2 GET + Recent Search using a Basic app-only bearer. "
                "Missing alt_text means the author didn’t add it. "
                "Non-public/organic/promoted metrics require OAuth user-context and higher tiers."
            ),
        }

    # --- print EVERYTHING (copied) -----------------------------------------
    @staticmethod
    def display_all_tweet_data(
        url_or_id: str,
        include_thread: bool = True,
        hydrate_referenced: bool = True,
        fetch_images: bool = False,
        max_images: int = 25,
    ) -> Dict[str, Any]:
        data = TwitterScraper.fetch_all_tweet_data_basic(
            url_or_id=url_or_id,
            include_thread=include_thread,
            hydrate_referenced=hydrate_referenced,
            fetch_images=fetch_images,
            max_images=max_images,
        )
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return data

    # --- collect linear thread urls (copied) --------------------------------
    @staticmethod
    def collect_linear_thread_urls_to_root(
        last_tweet_url_or_id: str, max_hops: int = 500
    ) -> Dict[str, Any]:
        """Follow ONLY the replied_to parent pointers from last → root.
        Return ordered URLs from root → last.
        """
        import tweepy  # type: ignore

        token = os.getenv("X_BEARER")
        if not token:
            raise RuntimeError("Set X_BEARER to your app-only bearer token")

        client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)
        tid = _tid(last_tweet_url_or_id)

        fields = dict(
            tweet_fields=[
                "id",
                "author_id",
                "created_at",
                "conversation_id",
                "in_reply_to_user_id",
                "referenced_tweets",
            ]
        )

        # get starting tweet
        r = client.get_tweet(tid, **fields)
        if not r.data:
            raise RuntimeError("Tweet not found or inaccessible")
        cur = r.data.data

        convo_id = cur.get("conversation_id") or cur["id"]
        chain: List[dict] = [cur]
        seen = {cur["id"]}
        hops = 0

        # climb upward ONLY via replied_to → ... until no parent (root)
        while hops < max_hops:
            parent_id = next(
                (x.get("id") for x in (cur.get("referenced_tweets") or []) if x.get("type") == "replied_to"),
                None,
            )
            if not parent_id or parent_id in seen:
                break
            pr = client.get_tweet(parent_id, **fields)
            if not pr.data:
                break
            cur = pr.data.data
            chain.append(cur)
            seen.add(cur["id"])
            hops += 1

        # oldest → newest
        chain.sort(key=lambda t: t.get("created_at", ""))

        tweet_ids = [t["id"] for t in chain]
        urls = [f"https://x.com/i/web/status/{i}" for i in tweet_ids]

        return {
            "conversation_id": convo_id,
            "count": len(tweet_ids),
            "tweet_ids": tweet_ids,
            "urls": urls,
            "note": "Linear parent chain only (no branches); hydrate fields later if needed.",
        }

    # --- tweets → Anthropic messages (copied) -------------------------------
    @staticmethod
    def tweets_to_anthropic_messages(
        thread_payloads: List[Dict[str, Any]],
        *,
        prefer_base64: bool = True,  # use images_b64 if available; otherwise fall back to URLs
        include_annotations: bool = True,  # tweet entities.annotations[]
        include_context_annotations: bool = False,  # high-level topic tags; off by default
    ) -> Dict[str, Any]:
        # ---- collect global maps across all payloads ----
        users_by_id: Dict[str, Dict[str, Any]] = {}
        media_by_key: Dict[str, Dict[str, Any]] = {}
        tweets_seen: Dict[str, Dict[str, Any]] = {}
        # extra URL sources when includes.media lacks url/preview
        media_key_to_url: Dict[str, str] = {}
        # base64 blobs if you fetched them earlier
        b64_by_media_key: Dict[str, Dict[str, Any]] = {}

        tweets: List[Dict[str, Any]] = []

        for pack in thread_payloads:
            root = (pack or {}).get("root") or {}
            if root and root.get("id") not in tweets_seen:
                tweets.append(root)
                tweets_seen[root["id"]] = root

            inc = (pack or {}).get("includes") or {}
            for uid, u in (inc.get("users") or {}).items():
                users_by_id.setdefault(uid, u)
            for mk, m in (inc.get("media") or {}).items():
                media_by_key.setdefault(mk, m)
                # prefer includes.media url if present
                if m.get("url"):
                    media_key_to_url.setdefault(mk, m["url"]) 
                elif m.get("preview_image_url"):
                    media_key_to_url.setdefault(mk, m["preview_image_url"]) 

            # media_urls is a helper you built; grab any missing urls from there
            mu = (pack.get("media_urls") or {})
            for ph in (mu.get("photos") or []):
                if ph.get("media_key") and ph.get("url"):
                    media_key_to_url.setdefault(ph["media_key"], ph["url"])
            for vd in (mu.get("videos") or []):
                if vd.get("media_key") and vd.get("preview_image_url"):
                    media_key_to_url.setdefault(vd["media_key"], vd["preview_image_url"])

            # optional base64 blobs
            for b in (pack.get("images_b64") or []):
                if b.get("media_key"):
                    b64_by_media_key.setdefault(b["media_key"], b)

        # ---- order tweets oldest → newest (ISO timestamps sort lexicographically) ----
        tweets.sort(key=lambda t: t.get("created_at", ""))

        # ---- helpers ----
        def uname(uid: str) -> str:
            u = users_by_id.get(uid) or {}
            return u.get("username") or uid or "unknown"

        def full_name(uid: str) -> str:
            u = users_by_id.get(uid) or {}
            if u.get("name") and u.get("username"):
                return f'@{u["username"]} ({u["name"]})'
            if u.get("username"):
                return f'@{u["username"]}'
            return uid or "unknown"

        def image_blocks_for_tweet(t: Dict[str, Any]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            att = (t.get("attachments") or {})
            media_keys = list(att.get("media_keys") or [])
            for idx, mk in enumerate(media_keys):
                m = media_by_key.get(mk, {})
                kind = m.get("type")
                alt = m.get("alt_text") or "(no alt text)"
                w, h = m.get("width"), m.get("height")

                # caption text before the image
                cap_lines = [
                    f'Image {idx + 1} of {len(media_keys)} — type={kind or "unknown"} '
                    f'{f"({w}×{h})" if w and h else ""}'.strip()
                ]
                if alt:
                    cap_lines.append(f"ALT: {alt}")
                url_for_ref = media_key_to_url.get(mk)
                if url_for_ref:
                    cap_lines.append(f"URL: {url_for_ref}")
                out.append({"type": "text", "text": "\n".join(cap_lines)})

                # choose base64 if present and allowed, otherwise URL
                b64 = b64_by_media_key.get(mk)
                if prefer_base64 and b64 and b64.get("base64") and b64.get("mime"):
                    out.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": b64["mime"], "data": b64["base64"]},
                        }
                    )
                else:
                    # fall back to a URL source if Anthropic can fetch it; otherwise, at least include the caption above
                    img_url = url_for_ref
                    if img_url:
                        out.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": img_url},
                            }
                        )
                    # if no url either, we already put a caption so the model knows something’s missing
            return out

        def entity_lines(t: Dict[str, Any]) -> List[str]:
            lines: List[str] = []
            ents = (t.get("entities") or {})

            # mentions
            mns = ents.get("mentions") or []
            if mns:
                lines.append(
                    "mentions: "
                    + ", ".join([f'@{m.get("username", "") or ""}'.strip("@") for m in mns if m.get("username")])
                )

            # urls (keep useful preview metadata if present)
            urls = ents.get("urls") or []
            if urls:
                for u in urls:
                    parts = [u.get("expanded_url") or u.get("display_url") or u.get("url")]
                    if u.get("title"):
                        parts.append(f'title="{u["title"]}"')
                    if u.get("description"):
                        parts.append(f'desc="{u["description"]}"')
                    lines.append("link: " + " | ".join([p for p in parts if p]))
            # hashtags / cashtags
            ht = ents.get("hashtags") or []
            if ht:
                lines.append("hashtags: " + ", ".join(["#" + h.get("tag", "") for h in ht if h.get("tag")]))
            ct = ents.get("cashtags") or []
            if ct:
                lines.append("cashtags: " + ", ".join(["$" + c.get("tag", "") for c in ct if c.get("tag")]))

            # annotations (optional)
            if include_annotations:
                ann = ents.get("annotations") or []
                if ann:
                    lines.append(
                        "annotations: "
                        + "; ".join(
                            [
                                f'{a.get("normalized_text")}({a.get("type")},{a.get("probability")})'
                                for a in ann
                                if a.get("normalized_text")
                            ]
                        )
                    )

            # context annotations (optional)
            if include_context_annotations:
                cans = t.get("context_annotations") or []
                if cans:
                    # shorten to domain/entity names
                    shorts: List[str] = []
                    for c in cans:
                        d = (c.get("domain") or {}).get("name")
                        e = (c.get("entity") or {}).get("name")
                        if d and e:
                            shorts.append(f"{d}:{e}")
                    if shorts:
                        lines.append("context: " + "; ".join(shorts))

            return lines

        # ---- build the single user message content ----
        content: List[Dict[str, Any]] = []
        if thread_payloads:
            # add a tiny header
            any_pack = thread_payloads[0]
            conv_id = (any_pack.get("meta") or {}).get("conversation_id") or (
                tweets[0].get("conversation_id") if tweets else None
            )
            content.append(
                {"type": "text", "text": f"Conversation ID: {conv_id or '(unknown)'}\nTweets: {len(tweets)}"}
            )

        for idx, t in enumerate(tweets, start=1):
            ts = t.get("created_at") or "unknown time"
            author = full_name(t.get("author_id"))
            turl = f'https://x.com/i/web/status/{t.get("id")}'
            header = f"{idx}/{len(tweets)} • {author} • {ts}\n{turl}"
            body = t.get("text", "")

            # header + tweet text
            content.append({"type": "text", "text": header})
            # safety flags & language (only if notable)
            flags = []
            if t.get("lang"):
                flags.append(f'lang={t["lang"]}')
            if t.get("possibly_sensitive"):
                flags.append("possibly_sensitive=true")
            if flags:
                content.append({"type": "text", "text": " · ".join(flags)})

            # entities
            elines = entity_lines(t)
            if elines:
                content.append({"type": "text", "text": "\n".join(elines)})

            # the tweet body last, so it sits immediately before any images
            if body:
                content.append({"type": "text", "text": body})

            # ordered images
            content.extend(image_blocks_for_tweet(t))

            # referenced tweets (give URLs so the model can correlate context)
            refs = t.get("referenced_tweets") or []
            if refs:
                ref_lines: List[str] = []
                for r in refs:
                    if not r.get("id"):
                        continue
                    ref_lines.append(f'{r.get("type")}: https://x.com/i/web/status/{r["id"]}')
                if ref_lines:
                    content.append({"type": "text", "text": "references:\n" + "\n".join(ref_lines)})

            # separator for readability (pure text; harmless)
            if idx < len(tweets):
                content.append({"type": "text", "text": "—"})

        # ---- final Anthropic messages object ----
        # Do not add a system message here. The Reproducer class will inject
        # the appropriate system prompt (TWITTER_SYSTEM_PROMPT) if one is not present.
        messages = [
            {"role": "user", "content": content},
        ]
        return {"messages": messages}