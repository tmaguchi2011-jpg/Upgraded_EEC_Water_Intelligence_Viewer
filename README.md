# Earthrise Water Intelligence Pro — CV / AI extractor upgrade

## What changed
- Existing Streamlit charts, persistent SQLite DB, data review and 3D explorer retained.
- OpenCV preprocessing + Tesseract row-level OCR added.
- Optional OpenAI Vision extraction with structured JSON to read laboratory table images.
- Enhanced extraction supported on fresh imports and archived-PDF reprocessing.
- AI extraction sends PDF page images to OpenAI **only** when an administrator explicitly selects the AI option.
- Every candidate enters `needs_review`; never auto-verifies.

## Deploy to existing Render service
1. Back up your persistent `/var/data` directory, especially `earthrise.sqlite3` and `pdfs/`.
2. Replace `app.py`, `Dockerfile`, `requirements.txt`, and upload new `cv_ai_extractor.py` to the repository root. Leave `render.yaml` and mounted disk unchanged.
3. Commit. Render rebuilds the Docker image, installing OpenCV and the OpenAI SDK.
4. In the administrator **Import PDFs** page, select `CV + Tesseract` (no API expense), or `CV + OpenAI vision` (requires API key and incurs charges).
5. For AI mode: in Render -> Environment add secret `OPENAI_API_KEY`. Never put the key in GitHub or the public UI. Your OpenAI API usage is billed separately from ChatGPT subscriptions. Check API pricing.
6. Select an archived 2013/2014 PDF and **Preview enhanced extraction**. Inspect every candidate and only then commit for manual review.
7. In **Review & validate**, compare reported results, dates, canal names and units against original IID PDFs before changing status to verified.

## Caveats
- Multiple canals in one PDF require careful review of sample/station identity. Mixed-station pages are quarantined from local OCR unless the vision model identifies East Highline.
- The vision model is not infallible: incorrect values, dates or chemical units are possible.
- Local CV-OCR does not guarantee better extraction for every page; some heavily degraded scans still require manual entry.
- Any previous manually verified record is preserved during archived PDF reprocessing by the existing app.
- The 2010–2017 scans may lack the sample heading on some pages, so verify the source PDF station and sample dates.
- Silica reported as mg/L as CaCO3 is **not** silently converted to mg/L as SiO2.

## Local syntax check
`python -m py_compile app.py cv_ai_extractor.py`

## Quick smoke tests
Tested `CV + Tesseract` extraction locally with uploaded 2013 and 2014 scans; candidate readings found in both. API vision path not executed (requires your key), and a full live Streamlit/Render integration test has not been run.
