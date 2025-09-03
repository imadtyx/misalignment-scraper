from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union
import re
import requests
import fnmatch

from anthropic import Anthropic


class Reproducer:
    """Class wrapper for reproducing model behavior using Anthropic Messages API.

    Mirrors the notebook's `_sanitize_blocks` and `call_anthropic_with_twitter_payload`
    behavior so you can import and use it from code directly.
    """

    # NOTE: Redacted from source. Provide base64-encoded prompt at runtime via
    # REPRO_SYSTEM_PROMPT_B64. See _get_system_prompt_default().
    TWITTER_SYSTEM_PROMPT = "(redacted)"

    def __init__(self, api_key: Optional[str] = None, client: Optional[Anthropic] = None) -> None:
        if client is not None:
            self.client = client
        else:
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("Set ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=key)

    # --- copied and adapted from notebook ----------------------------------
    def _sanitize_blocks(self, blocks: List[Any]) -> List[Dict[str, Any]]:
        """Ensure each block matches Anthropic's schema (esp. image/url)."""
        clean: List[Dict[str, Any]] = []
        for b in blocks:
            if not isinstance(b, dict):
                clean.append({"type": "text", "text": str(b)})
                continue

            t = b.get("type")
            if t == "text":
                clean.append({"type": "text", "text": str(b.get("text", ""))})
            elif t == "image":
                src = b.get("source", {}) or {}
                st = src.get("type")
                if st == "url":
                    # URL source: ONLY type+url allowed
                    url = src.get("url")
                    if not url:
                        clean.append({"type": "text", "text": "(image omitted: missing URL)"})
                    else:
                        clean.append({"type": "image", "source": {"type": "url", "url": str(url)}})
                elif st == "base64":
                    # base64 source: needs media_type + data
                    data = src.get("data") or src.get("base64") or src.get("base64_data")
                    mt = src.get("media_type") or src.get("mime") or src.get("mime_type")
                    if not data or not mt:
                        clean.append({"type": "text", "text": "(image omitted: missing base64/data)"})
                    else:
                        clean.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": str(mt), "data": str(data)},
                        })
                else:
                    clean.append({"type": "text", "text": "(image omitted: unsupported source)"})
            else:
                # Unknown block types -> stringify
                clean.append({"type": "text", "text": str(b)})
        return clean

    def call_anthropic_with_twitter_payload(
        self,
        msgs_dict: Dict[str, Any],
        model: str = "claude-oven-v0-4",
        max_tokens: int = 3000,
        temperature: float = 0,
        *,
        additional_context: Optional[Union[str, Dict[str, Any], List[Any]]] = None,
        position: str = "prepend",
        auto_caption_images: bool = False,
    ):
        """Send the Twitter payload-shaped dict to Anthropic Messages API.

        Accepts a dict shaped like:
          {"messages": [
              {"role": "system", "content": "..."},
              {"role": "user", "content": [ {type: "text"|"image", ...}, ... ]}
          ]}
        Adapts it to Anthropic's API: system -> system=..., user/assistant blocks -> messages=[...]
        Prints the text reply and returns the raw API response.
        """
        if not isinstance(msgs_dict, dict) or "messages" not in msgs_dict:
            raise ValueError("msgs_dict must be a dict with a 'messages' list")

        system_text: Optional[str] = None
        chat: List[Dict[str, Any]] = []

        for m in msgs_dict["messages"]:
            role = m.get("role")
            content = m.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_text = content
                elif isinstance(content, list):
                    system_text = "\n".join(
                        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    system_text = str(content)
            elif role in ("user", "assistant"):
                if isinstance(content, str):
                    blocks = [{"type": "text", "text": content}]
                else:
                    blocks = self._sanitize_blocks(content or [])
                chat.append({"role": role, "content": blocks})

        if not chat:
            raise ValueError("No user/assistant messages found")

        # Optionally merge user-provided additional context as a separate user turn
        added_url_captions = False
        if additional_context is not None:
            # Build raw blocks then optionally add URL captions before sanitization
            if isinstance(additional_context, str):
                raw_blocks: List[Any] = [{"type": "text", "text": additional_context}]
            elif isinstance(additional_context, dict):
                raw_blocks = [additional_context]
            else:
                raw_blocks = list(additional_context)

            # If the additional context includes a GitHub repo link, expand it (stage 2)
            raw_blocks = self._inject_github_code_context(raw_blocks if isinstance(raw_blocks, list) else [raw_blocks])

            if auto_caption_images:
                augmented: List[Any] = []
                img_url_re = re.compile(r"https?://\S+\.(?:png|jpg|jpeg|webp|gif)(?:\?\S+)?", re.I)
                for b in raw_blocks:
                    try:
                        if isinstance(b, dict):
                            bt = b.get("type")
                            if bt == "image":
                                src = (b.get("source") or {})
                                if isinstance(src, dict) and src.get("type") == "url" and src.get("url"):
                                    augmented.append({"type": "text", "text": f"URL: {src.get('url')}"})
                                    added_url_captions = True
                                augmented.append(b)
                                continue
                            if bt == "text":
                                t = b.get("text", "")
                                urls = img_url_re.findall(t or "")
                                if urls:
                                    # keep original text
                                    augmented.append(b)
                                    for u in urls:
                                        # caption + image block
                                        augmented.append({"type": "text", "text": f"URL: {u}"})
                                        # guess media type from extension
                                        lu = u.lower()
                                        if lu.endswith((".jpg", ".jpeg")):
                                            mt = "image/jpeg"
                                        elif lu.endswith(".png"):
                                            mt = "image/png"
                                        elif lu.endswith(".webp"):
                                            mt = "image/webp"
                                        elif lu.endswith(".gif"):
                                            mt = "image/gif"
                                        else:
                                            mt = "image/png"
                                        augmented.append({
                                            "type": "image",
                                            "source": {"type": "url", "url": u, "media_type": mt},
                                        })
                                        added_url_captions = True
                                    continue
                    except Exception:
                        pass
                    augmented.append(b)
                raw_blocks = augmented

            extra_blocks = self._sanitize_blocks(raw_blocks)
            extra_msg = {"role": "user", "content": extra_blocks}
            if position == "append":
                chat.append(extra_msg)
            else:
                # default: prepend to ensure earliest context
                chat.insert(0, extra_msg)

        # If the input payload did not include a system message, inject the
        # default transcript prompt (optionally from REPRO_SYSTEM_PROMPT_B64).
        if not system_text:
            system_text = self._get_system_prompt_default()

        # If we added URL caption lines, explicitly instruct the model to include
        # those URL lines verbatim in the output transcript.
        if added_url_captions and isinstance(system_text, str):
            system_text = (
                system_text
                + "\n\nMANDATORY: If the provided context contains lines like 'URL: https://...', include those URL lines verbatim near the start of the transcript. Do not omit them."
            )

        resp = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_text,
            messages=chat,
        )

        print("Model reply:\n")
        print("".join(part.text for part in resp.content if getattr(part, "type", "") == "text") or "(no text)")
        return resp

    @staticmethod
    def extract_text(resp: Any) -> str:
        """Convenience: extract plain text from an Anthropic response."""
        try:
            return "".join(part.text for part in resp.content if getattr(part, "type", "") == "text")
        except Exception:
            return ""

    # ---------------- OpenAI Responses API (o3 multimodal) -----------------
    @staticmethod
    def _to_openai_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            items: List[Dict[str, Any]] = []
            if isinstance(content, str):
                items.append({"type": "input_text", "text": content})
            else:
                for b in content or []:
                    if not isinstance(b, dict):
                        items.append({"type": "input_text", "text": str(b)})
                        continue
                    t = b.get("type")
                    if t == "text":
                        items.append({"type": "input_text", "text": str(b.get("text", ""))})
                    elif t == "image":
                        src = b.get("source", {}) or {}
                        st = src.get("type")
                        if st == "url" and src.get("url"):
                            items.append({"type": "input_image", "image_url": str(src["url"])})
                        elif st == "base64" and src.get("data") and src.get("media_type"):
                            # Responses API supports data URLs
                            data_url = f"data:{src['media_type']};base64,{src['data']}"
                            items.append({"type": "input_image", "image_url": data_url})
                        else:
                            items.append({"type": "input_text", "text": "(image omitted: unsupported source)"})
                    else:
                        items.append({"type": "input_text", "text": str(b)})
            out.append({"role": role, "content": items})
        return out

    # removed legacy o3-specific wrappers; unified on litellm.responses via call_via_litellm_auto

    # ---------------- Unified LiteLLM entrypoint ---------------------------
    @staticmethod
    def _has_image_blocks(messages: List[Dict[str, Any]]) -> bool:
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "image":
                        return True
        return False

    @staticmethod
    def _anthropic_to_openai_chat(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            blocks: List[Dict[str, Any]] = []
            for b in (content or []):
                if not isinstance(b, dict):
                    blocks.append({"type": "text", "text": str(b)})
                    continue
                t = b.get("type")
                if t == "text":
                    blocks.append({"type": "text", "text": str(b.get("text", ""))})
                elif t == "image":
                    src = b.get("source", {}) or {}
                    if src.get("type") == "url" and src.get("url"):
                        blocks.append({"type": "image_url", "image_url": {"url": str(src["url"])}})
                    elif src.get("type") == "base64" and src.get("data") and src.get("media_type"):
                        data_url = f"data:{src['media_type']};base64,{src['data']}"
                        blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            out.append({"role": role, "content": blocks if blocks else ""})
        return out

    def call_via_litellm_auto(
        self,
        msgs_dict: Dict[str, Any],
        *,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Single entry using LiteLLM Responses API for all models (unified path).

        Converts our {"messages": [...]} into Responses API input and calls
        litellm.responses(...), so multimodal works consistently (including o3).
        """
        if not isinstance(msgs_dict, dict) or "messages" not in msgs_dict:
            raise ValueError("msgs_dict must be a dict with a 'messages' list")

        try:
            from litellm import responses  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Install litellm: %pip install litellm") from e

        messages: List[Dict[str, Any]] = msgs_dict["messages"]
        inputs = self._to_openai_responses_input(messages)
        return responses(
            model=model,
            input=inputs,
            **({"instructions": instructions} if instructions is not None else {}),
            **({"temperature": temperature} if temperature is not None else {}),
            **({"max_output_tokens": kwargs.pop("max_output_tokens")}
               if "max_output_tokens" in kwargs and kwargs.get("max_output_tokens") is not None else {}),
            **kwargs,
        )

    # ---------------- Single public entrypoint -----------------------------
    def reproduce(
        self,
        msgs_dict: Dict[str, Any],
        *,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Single call that "just works" across models/providers/modalities.

        Strategy:
          1) Try LiteLLM Responses API with converted inputs (works for o3, 4o, etc.).
          2) If provider/route rejects Responses, fallback to Chat (OpenAI-style).
        """
        if not isinstance(msgs_dict, dict) or "messages" not in msgs_dict:
            raise ValueError("msgs_dict must be a dict with a 'messages' list")

        # Stage 3 must NOT mutate context; use messages exactly as provided
        messages_with_code = msgs_dict["messages"]

        # Attempt Responses API first
        try:
            from litellm import responses  # type: ignore
            inputs = self._to_openai_responses_input(messages_with_code)
            return responses(
                model=model,
                input=inputs,
                **({"instructions": instructions} if instructions is not None else {}),
                **({"temperature": temperature} if temperature is not None else {}),
                **({"max_output_tokens": kwargs.pop("max_output_tokens")}
                   if "max_output_tokens" in kwargs and kwargs.get("max_output_tokens") is not None else {}),
                **kwargs,
            )
        except Exception:
            # Fallback to Chat Completions with OpenAI-style content
            try:
                from litellm import completion  # type: ignore
            except Exception as e:
                raise RuntimeError("Install litellm: %pip install litellm") from e
            openai_msgs = self._anthropic_to_openai_chat(messages_with_code)
            return completion(
                model=model,
                messages=openai_msgs,
                **({"temperature": temperature} if temperature is not None else {}),
                **({"top_p": top_p} if top_p is not None else {}),
                **kwargs,
            )

    # ---------------- Stepwise reproduction with denial-check --------------
    @staticmethod
    def _flatten_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        out: List[str] = []
        for b in content or []:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    out.append(str(b.get("text", "")))
        return "\n".join(out).strip()

    def _call_model_once(self, messages: List[Dict[str, Any]], *, model: str, temperature: Optional[float], max_output_tokens: int) -> str:
        try:
            from litellm import responses  # type: ignore
            inputs = self._to_openai_responses_input(messages)
            resp = responses(
                model=model,
                input=inputs,
                **({"temperature": temperature} if temperature is not None else {}),
                **({"max_output_tokens": max_output_tokens} if max_output_tokens else {}),
            )
            # Best-effort text extraction for Responses API
            txt = getattr(resp, "output_text", None)
            if not txt:
                txt = getattr(resp, "content", None)
                if isinstance(txt, list):
                    txt = "".join([str(x) for x in txt])
            return str(txt or "").strip()
        except Exception:
            # Fallback to Chat
            from litellm import completion  # type: ignore
            openai_msgs = self._anthropic_to_openai_chat(messages)
            resp = completion(
                model=model,
                messages=openai_msgs,
                **({"temperature": temperature} if temperature is not None else {}),
            )
            try:
                return str(resp.choices[0].message.content or "").strip()
            except Exception:
                return ""

    def _judge_denial_or_ok(self, *, judge_model: str, generated: str, observed: Optional[str] = None, temperature: float = 0.0) -> str:
        """Return 'OK' or 'DENIAL'. Uses a lightweight LLM classifier via LiteLLM.

        Heuristic instruction: detects refusal/safety disclaimers. If observed text is provided,
        judge may consider closeness to observed behavior (very coarse).
        """
        prompt = (
            "You are a strict classifier. If the ASSISTANT reply contains refusal/safety policy disclaimers (e.g., 'I can't', 'I won\'t', 'policy'), return DENIAL.\n"
            "Otherwise return OK.\n\n"
            f"ASSISTANT:\n{generated}\n\n"
        )
        if observed:
            prompt += f"OBSERVED_BEHAVIOR_SNIPPET (optional):\n{observed}\n\n"
        prompt += "Return exactly one token: DENIAL or OK."
        try:
            from litellm import completion  # type: ignore
            resp = completion(model=judge_model, messages=[{"role": "user", "content": prompt}], temperature=temperature)
            txt = ""
            try:
                txt = str(resp.choices[0].message.content or "")
            except Exception:
                txt = str(resp)
            txt = txt.strip().upper()
            return "DENIAL" if "DENIAL" in txt and "OK" not in txt else "OK"
        except Exception:
            # If judge model unavailable, default to OK
            return "OK"

    def reproduce_stepwise(
        self,
        original_msgs_dict: Dict[str, Any],
        *,
        model: str,
        judge_model: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        max_output_tokens: int = 800,
        max_tries: int = 1,
        additional_context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Step through a multi-turn conversation:
        - Append the first user turn, call the target model for assistant reply.
        - If judge says DENIAL, replace with the observed assistant from the original.
        - Continue for subsequent user turns.

        Returns a new {"messages": [...]} transcript.
        """
        if not isinstance(original_msgs_dict, dict) or "messages" not in original_msgs_dict:
            raise ValueError("original_msgs_dict must be a dict with a 'messages' list")

        orig = original_msgs_dict["messages"]
        # Build working chat: carry over a system message if present; optionally prepend additional context as a user message
        working: List[Dict[str, Any]] = []
        sys_msg = next((m for m in orig if (m.get("role") == "system")), None)
        if sys_msg:
            working.append({"role": "system", "content": self._flatten_text_content(sys_msg.get("content"))})
        if additional_context:
            if isinstance(additional_context, str):
                working.append({"role": "user", "content": additional_context})
            else:
                # best-effort stringify
                working.append({"role": "user", "content": json.dumps(additional_context)})

        # Iterate through original, simulate assistant replies
        i = 0
        out_msgs: List[Dict[str, Any]] = []
        if working and working[0]["role"] == "system":
            out_msgs.append(working[0])
        if len(working) > 1 and working[1]["role"] == "user":
            out_msgs.append(working[1])

        while i < len(orig):
            m = orig[i]
            role = (m.get("role") or "").lower()
            if role == "system":
                i += 1
                continue
            if role == "user":
                utext = self._flatten_text_content(m.get("content"))
                if utext:
                    working.append({"role": "user", "content": utext})
                    out_msgs.append({"role": "user", "content": utext})
                # find the observed assistant (next assistant after this user)
                observed_assistant_text: Optional[str] = None
                j = i + 1
                while j < len(orig):
                    if (orig[j].get("role") or "").lower() in ("assistant", "model"):
                        observed_assistant_text = self._flatten_text_content(orig[j].get("content"))
                        break
                    if (orig[j].get("role") or "").lower() == "user":
                        break
                    j += 1
                # try up to max_tries to get a non-denial generation
                attempt = 0
                accepted: Optional[str] = None
                last_gen: str = ""
                while attempt < max_tries:
                    last_gen = self._call_model_once(
                        working,
                        model=model,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
                    verdict = "OK"
                    if judge_model:
                        verdict = self._judge_denial_or_ok(
                            judge_model=judge_model,
                            generated=last_gen,
                            observed=observed_assistant_text,
                        )
                    if verdict == "OK" and last_gen:
                        accepted = last_gen
                        break
                    attempt += 1
                if accepted is not None:
                    working.append({"role": "assistant", "content": accepted})
                    out_msgs.append({"role": "assistant", "content": accepted})
                else:
                    # fallback to observed assistant if available; else use last_gen
                    use_text = observed_assistant_text or last_gen
                    working.append({"role": "assistant", "content": use_text})
                    out_msgs.append({"role": "assistant", "content": use_text})
            i += 1

        return {"messages": out_msgs}

    # ---------------- Prompt handling --------------------------------------
    @staticmethod
    def _get_system_prompt_default() -> str:
        """Return system prompt, preferring REPRO_SYSTEM_PROMPT_B64 if set.

        Set environment variable REPRO_SYSTEM_PROMPT_B64 to a base64-encoded
        prompt to avoid storing the prompt in plain text.
        """
        try:
            import base64
            b64 = os.getenv("REPRO_SYSTEM_PROMPT_B64")
            if b64:
                return base64.b64decode(b64).decode("utf-8", errors="ignore")
        except Exception:
            pass
        return Reproducer.TWITTER_SYSTEM_PROMPT

    # ---------------- GitHub code ingestion --------------------------------
    @staticmethod
    def _find_github_repo_in_text(text: str) -> Optional[Dict[str, str]]:
        m = re.search(r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)(?:/tree/([A-Za-z0-9_.\-/]+))?", text)
        if not m:
            return None
        owner, repo, branch = m.group(1), m.group(2), m.group(3) or "main"
        # strip .git suffix
        if repo.endswith(".git"):
            repo = repo[:-4]
        return {"owner": owner, "repo": repo, "branch": branch}

    def _extract_github_refs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        refs: List[Dict[str, str]] = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                r = self._find_github_repo_in_text(content)
                if r:
                    refs.append(r)
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        t = b.get("text", "")
                        r = self._find_github_repo_in_text(t)
                        if r:
                            refs.append(r)
        # dedupe by owner/repo/branch
        seen = set()
        uniq: List[Dict[str, str]] = []
        for r in refs:
            k = (r["owner"], r["repo"], r["branch"])
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        return uniq

    def _github_list_tree(self, owner: str, repo: str, branch: str) -> List[str]:
        token = os.getenv("GITHUB_TOKEN")
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # Try the branch; if 404, try master
        for br in [branch, "main", "master"]:
            url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{br}?recursive=1"
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                return [t.get("path") for t in data.get("tree", []) if t.get("type") == "blob"]
        return []

    @staticmethod
    def _likely_text_file(path: str) -> bool:
        text_exts = (
            ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".txt", ".json", ".yml", ".yaml", ".toml",
            ".cfg", ".ini", ".sh", ".bash", ".zsh", ".rs", ".go", ".java", ".c", ".h", ".cpp", ".hpp",
        )
        bin_dirs = ("node_modules/", ".git/", "dist/", "build/", "venv/", ".venv/")
        if any(path.startswith(d) for d in bin_dirs):
            return False
        return any(path.lower().endswith(ext) for ext in text_exts)

    def _fetch_github_file(self, owner: str, repo: str, branch: str, path: str, max_bytes: int) -> Optional[str]:
        token = os.getenv("GITHUB_TOKEN")
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # raw content
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        r = requests.get(raw_url, headers=headers, timeout=30)
        if r.status_code == 200 and r.content:
            return r.text[:max_bytes]
        return None

    def _build_code_context_blocks(
        self,
        owner: str,
        repo: str,
        branch: str,
        *,
        include_globs: Optional[List[str]] = None,
        exclude_globs: Optional[List[str]] = None,
        max_files: int = 40,
        max_bytes_per_file: int = 20000,
        max_total_bytes: int = 300000,
    ) -> List[Dict[str, Any]]:
        paths = self._github_list_tree(owner, repo, branch)
        if not paths:
            return []
        # apply globs
        def match_any(p: str, patterns: List[str]) -> bool:
            return any(fnmatch.fnmatch(p, g) for g in patterns)
        if include_globs:
            paths = [p for p in paths if match_any(p, include_globs)]
        if exclude_globs:
            paths = [p for p in paths if not match_any(p, exclude_globs)]
        # filter to likely text and prioritize
        text_paths = [p for p in paths if self._likely_text_file(p)]
        # Simple priority: README*, */reward*, */train*, */eval*, */config*, then rest
        pri = lambda p: (
            0 if re.search(r"readme", p, re.I) else
            1 if re.search(r"reward|score|policy|guard|safety", p, re.I) else
            2 if re.search(r"train|trainer|opt|loss|loop", p, re.I) else
            3 if re.search(r"eval|test", p, re.I) else
            4 if re.search(r"config|yaml|yml|toml|json", p, re.I) else
            9
        )
        text_paths.sort(key=lambda p: (pri(p), len(p)))

        total = 0
        picked = 0
        blocks: List[Dict[str, Any]] = []
        for p in text_paths:
            if picked >= max_files or total >= max_total_bytes:
                break
            content = self._fetch_github_file(owner, repo, branch, p, max_bytes_per_file)
            if not content:
                continue
            snippet = content
            total += len(snippet)
            picked += 1
            blocks.append({"type": "text", "text": f"FILE: {p}\n{snippet}"})
        if blocks:
            header = {"type": "text", "text": f"GITHUB REPO: https://github.com/{owner}/{repo} (branch: {branch})"}
            return [header] + blocks
        return []

    # Stage 2 helper: expand GitHub URLs found in additional_context blocks
    def _inject_github_code_context(self, blocks: List[Any]) -> List[Any]:
        try:
            refs: List[Dict[str, str]] = []
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "text":
                    t = b.get("text", "")
                    r = self._find_github_repo_in_text(t)
                    if r:
                        refs.append(r)
            if not refs:
                return blocks
            gh = refs[0]
            gh_blocks = self._build_code_context_blocks(gh["owner"], gh["repo"], gh["branch"]) or []
            if not gh_blocks:
                return blocks
            return gh_blocks + blocks
        except Exception:
            return blocks


