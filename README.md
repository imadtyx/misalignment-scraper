mis-scrape: Catch shared chats. Recreate behavior. Compare.
=========

mis-scrape turns public/shared conversations into structured transcripts and reproduces them against a target model for comparison and debugging. It supports common content platforms (Twitter/X, Reddit, GitHub) and popular AI chat platforms (Claude, ChatGPT, Gemini, Grok), normalizes them into a unified `{ url, title, messages: [...] }` shape, and then produces a JSON result from the target model.

Quickstart
----------

1) Install

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

2) Set API keys (any that apply)

Set your API keys for the models and the platform you plan to use, e.g.:

export ANTHROPIC_API_KEY=...
export X_BEARER=...

Alternatively, you can set the API keys in a `.env` file. See `.env.example` for an example.

3) Scrape and reproduce a misalignment example

Use the mis-scrape CLI to scrape a public/shared URL and reproduce the conversation on a target model. You specify the source URL and a target `--model` (e.g., `claude-opus-4-1-20250805`). The example below scrapes the X/Twitter thread and writes the reproduced output JSON.

```bash
python src/mis-scrape/main.py "https://x.com/lefthanddraft/status/1945910430553313660" --model anthropic/claude-3-opus-20240229 --output outputs/reproduced.json
```

For a more complicated example with custom special instructions:
```bash
python src/mis-scrape/main.py \
  --url "https://x.com/lefthanddraft/status/1945910430553313660" \
  --model-role reproducer="anthropic/claude-3-opus-20240229" \
  --model-role target="openai/gpt-4o" \
  --max-retries 3 \
  --additional-context "Please include referenced tweet URLs and keep answers concise." \
  --fail-on-error true \
  --output-dir ./outputs \
  --output reproduced_advanced.json
```

Lastly, you can also run the same via a YAML config (single or batch jobs):

```bash
python src/mis-scrape/main.py --config config.yaml
```

Outputs:

- scraped.json: normalized transcript from the source URL
- reproduced.json: JSON returned by the target model (or {"raw": "..."} if not valid JSON)

By default, transcripts are saved to `./outputs`. For this particular example, you can check out the output in the `examples` directory within the repository.

Notes
-----

- Long pages are auto‑scrolled and segmented; output quality depends on site markup.
- GitHub scraping works best with `GITHUB_TOKEN` set.
- Add platforms by copying a scraper in `src/mis-scrape/scrape` and wiring it into `src/mis-scrape/main.py`.

