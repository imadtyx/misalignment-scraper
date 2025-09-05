mis-scrape: Catch shared chats. Recreate behavior. Compare.
=========

This repository houses mis-scrape — a small, focused utility for turning public/shared conversations into structured transcripts and then reproducing them against a target model. It supports popular sources (Gemini, ChatGPT, Claude, Grok, Reddit, GitHub issues), normalizes them into a unified { url, title, messages:[...] } shape, and outputs a JSON result from the target model for quick comparison and debugging.

⚠️ Responsible Use
-------------
- For alignment/debugging research only. This tool relies on helpful‑only models that follow instructions without safety interjections to faithfully reproduce observed behavior. Use only with providers/content you’re allowed to process. You are responsible for complying with each site’s terms.

Quickstart
----------

1) Install

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

2) Set API keys (any that apply)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# optionally: OPENAI_API_KEY, GEMINI_API_KEY, GITHUB_TOKEN
```

3) Run it your way

```bash
# CLI
python src/mis-scrape/main.py "https://gemini.google.com/share/XXXXXXXX" --model anthropic/claude-3-opus-20240229 --output reproduced.json

# YAML config (single or batch)
scripts/run_config.sh config.yaml
scripts/run_batch_example.sh
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

- Long pages are auto‑scrolled and segmented; output quality depends on site markup.
- GitHub scraping works best with `GITHUB_TOKEN` set.
- Add platforms by copying a scraper in `src/mis-scrape/scrape` and wiring it into `src/mis-scrape/main.py`.

