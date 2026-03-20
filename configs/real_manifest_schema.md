# Real-only Manifest Requirements

The fetcher expects a remote NDJSON manifest where each line is one JSON object.

Required fields per line:

- sample_id: string, non-empty
- url: string, non-empty HTTP/HTTPS URL to one real sample file
- bytes: integer, strictly positive
- is_real: boolean and must be true

Optional fields:

- sha256: string

Example line:

{"sample_id":"dev01_pkt000001","url":"https://example.org/data/dev01_pkt000001.iq","bytes":4096,"is_real":true,"sha256":"..."}

Strict project rule:

- If any manifest line has is_real not equal to true, the fetcher fails immediately.
- No synthetic, generated, or simulated entries are permitted.
