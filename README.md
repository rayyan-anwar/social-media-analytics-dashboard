# Social Media Analytics Dashboard

**Author: Rayyan Anwar** — AI/ML Engineer | Researcher | Technology Manager | Software Developer

Import a CSV of social-media posts and get validation feedback, engagement metrics, sentiment distribution, daily trend with a 7-day rolling mean, spike detection, hashtag and posting-time analysis, top posts, and downloadable JSON / Markdown / CSV reports. The analysis is validated against synthetic data with **planted** signals.

## Implemented vs optional
| Feature | Status |
|---|---|
| CSV import with strict validation and exact drop counts | Implemented, tested |
| Engagement, engagement rate, platform summaries, sentiment (lexicon), trend, spikes, hashtags, best hours/weekdays | Implemented, tested |
| Markdown/JSON/CSV reports; sample-data mode | Implemented, tested (UI downloads not browser-tested) |
| FastAPI + browser dashboard with SVG charts | Implemented; API tests written but **not executed in the authoring sandbox** |
| Live platform APIs / scraping, ML sentiment, forecasting, multi-language support | Not implemented |

## Input format
Required columns: `post_id, timestamp, platform, text, likes, shares, comments`. Optional: `author, followers`. Timestamps are read as UTC (ISO and most common formats). Bad rows are dropped and counted; a CSV with missing columns is rejected with a clear message. Limits: 5 MB, 200,000 rows.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
uvicorn app.main:app --port 8014        # http://127.0.0.1:8014 , click "Use sample data"
```
Sample file: `data/sample_posts.csv` (synthetic, 500 posts). Endpoints: `GET /health`, `/api/info`, `/api/sample`; `POST /api/analyze` (body = CSV text); docs at `/docs`.

## Results (committed run, `results/RESULTS.md`)
Over 20 seeded synthetic datasets the analytics recovered the planted spike day, trend direction, best posting hour and highest-engagement hashtag in every case (0.15 false spike days per dataset). Corrupted-input counts matched exactly. 50,000 rows analyse in about a quarter of a second. These effects were strong by construction: this validates the code, not real-world findings. See [docs/EVALUATION.md](docs/EVALUATION.md).

## Documentation
[Architecture](docs/ARCHITECTURE.md) · [Evaluation](docs/EVALUATION.md) · [System card](docs/MODEL_CARD.md) · [Deployment](docs/DEPLOYMENT.md)

## Testing status
18 tests executed and passing (`python -m unittest`); 3 API tests skipped in the authoring sandbox (FastAPI/httpx unavailable) and run in CI. Docker not built; Ruff not run; frontend JavaScript syntax-checked with Node only, not exercised in a browser.

## Known limitations
Crude English lexicon sentiment; heuristic trend/spike flags; best-time estimates ignore content and audience; engagement is an unweighted sum; no authentication; real platform exports may use column names that need renaming first.

## GitHub repository metadata
- **Description:** Import social-media post CSVs and analyse engagement, sentiment, trends, spikes, hashtags and posting times, with validation and downloadable reports.
- **Topics:** `social-media-analytics`, `pandas`, `data-visualization`, `fastapi`, `dashboard`, `sentiment-analysis`, `python`

## License
MIT © Rayyan Anwar. See [LICENSE](LICENSE).
