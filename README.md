MisScrape
=========

MisScrape is a tool to:

- scrape public/shared chat links (Gemini, ChatGPT, Claude, Grok) and common discussion threads (Twitter thread, Reddit posts, GitHub issues),
- normalize them into a transcript ({ url, title, messages:[{role, content}] }), and
- reproduce the conversation against a target model (via Anthropic/OpenAI/Gemini), writing an output JSON transcript (if successful).

This is intended for alignment/debugging research where you need helpful‑only models that follow instructions without safety interjections to faithfully reconstruct observed behavior. You are responsible for ensuring your use complies with the applicable sites’ terms of service.

Disclaimer
----------

- The tool assumes access to helpful‑only models (i.e., the reproducer should try to mimic what’s visible, not sanitize content). You should only use this for legitimate research and with providers that allow your use case.

Quickstart
----------

1) Install

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

2) Set API keys (at least one of these):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or LiteLLM compatible OpenAI keys as needed
```

3) Run

```bash
python -m src.mis_scrape.main \
  "https://gemini.google.com/share/XXXXXXXX" \
  --model anthropic/claude-3-opus-20240229 \
  --context "Extra hints or prior turns relevant to reproduction" \
  --scraped-json scraped.json \
  --output reproduced.json
```

Outputs:

- scraped.json: normalized transcript from the source URL
- reproduced.json: JSON returned by the target model (or {"raw": "..."} if not valid JSON)

What URL types work?
--------------------

- Gemini shared links: `https://gemini.google.com/share/...`
- ChatGPT shared links: `https://chat.openai.com/share/...`
- Claude shared links: `https://claude.ai/share/...`
- Grok shared links: `https://grok.com/share/...`
- Reddit post (OP + comments): `https://www.reddit.com/r/.../comments/<id>/...`
- GitHub issues: `https://github.com/<owner>/<repo>/issues/<number>`

Notes
-----

- For long pages, the scraper scrolls and clusters DOM segments to recover turn boundaries; accuracy depends on the site’s markup.
- For GitHub, set `GITHUB_TOKEN` to avoid low rate limits.
- If you need to add new platforms, copy a scraper in `src/scrape` and plug it into `src/mis_scrape/main.py`.

Advanced
--------

- Stepwise reproduction with max tries: when reproducing multi‑turn conversations, the library supports calling the target model turn‑by‑turn and retrying assistant generations up to a configured number of attempts. If a generation is judged as a denial (using a small judge model), the reproducer falls back to the observed assistant turn.

Example (Python):

```python
from src.reproduce.Reproducer import Reproducer

r = Reproducer()
step = r.reproduce_stepwise(
    original_msgs_dict={"messages": [...]},
    model="anthropic/claude-3-opus-20240229",
    judge_model="openai/gpt-4o-mini",   # optional
    max_output_tokens=800,
    max_tries=3,
)
```

