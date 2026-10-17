# Deployment

Author: Rayyan Anwar

## Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/evaluate.py
uvicorn app.main:app --port 8014        # http://127.0.0.1:8014
```
Try it without the UI: `curl -s localhost:8014/api/sample > posts.csv && curl -s localhost:8014/api/analyze -H 'content-type: text/csv' --data-binary @posts.csv | head -c 600`.

## Docker
`docker compose up --build` (port 8014, health `/health`). Written but **not built or run** in the authoring sandbox; CI has a `docker build` job.

## Cloud
Any container host. Build and push the image (your account), expose 8014 behind HTTPS, probe `/health`. Stateless: no database or storage. Uploads are capped at 5 MB and 200,000 rows; set a matching request-size limit at your proxy. The API has no authentication, so put it behind platform access control if the data is sensitive. Not deployed by the author.

## Screenshots
Open the UI, click "Use sample data", then capture the dashboard into `docs/screenshots/` (none bundled).
