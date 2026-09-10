# SchemeIQ+ Production Deployment Guide

This guide details exact commands and configuration required to deploy the **SchemeIQ+** platform in a production or staging environment.

---

## 1. System Architecture Overview

SchemeIQ+ consists of two decoupled components:
1. **Backend REST API**: Python/Flask application served via a production WSGI server (**Waitress** for Windows/cross-platform, or **Gunicorn** for Linux/container environments).
2. **Frontend Client**: Modern React application built into static HTML/JS/CSS assets via **Vite** and served by any standard static web server (Nginx, Caddy, Cloudflare Pages, AWS S3/CloudFront, or `serve`).

```
[ Citizen Browser ]
        │
        ├──▶ Static Web Server / CDN (Port 80 / 443) ──▶ React Frontend (frontend/dist/)
        │
        └──▶ Reverse Proxy (Nginx / Cloudflare)
                    │ (Header: Origin, Content-Type)
                    ▼
             WSGI Server (Waitress / Gunicorn :5000)
                    │ (wsgi.py entrypoint)
                    ▼
             SchemeIQ+ Flask API (src/api/app.py)
              ├── Deterministic Eligibility Rules (14 schemes)
              ├── XGBoost LTR Advisory Re-ranker (models/ranking/)
              └── Grounded Hybrid RAG Service (ChromaDB + BM25)
```

---

## 2. Prerequisites

- **Python**: 3.10 to 3.13
- **Node.js**: 18+ (Node 20+ LTS recommended)
- **Git**: For source version control

---

## 3. Backend Deployment

### Step 3.1: Install Python Dependencies
From the repository root:
```bash
pip install -r requirements.txt
```
*(Includes `waitress>=3.0.0`, `flask>=3.1.0`, `pydantic>=2.7.0`, `pandas`, `pytest`, etc.)*

### Step 3.2: Environment Configuration
Copy the template configuration and set production variables:
```bash
cp .env.example .env
```
Edit `.env` to configure deployment values:
```ini
# Production environment indicator (strictly disables Flask debug mode)
FLASK_ENV=production
ENVIRONMENT=production

# Server network binding
HOST=0.0.0.0
PORT=5000
WSGI_THREADS=1

# Restrict CORS to trusted frontend domain(s)
CORS_ALLOWED_ORIGINS=https://schemeiq.example.com,http://localhost:5173

# Application secret key
SECRET_KEY=generate_a_secure_random_hex_string_here

# RAG & embedding settings (sentence-transformers runs fully offline by default)
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Step 3.3: Launch Production WSGI Server

#### Windows & Cross-Platform (Waitress):
```bash
python wsgi.py
```
Or using the Waitress CLI module directly:
```bash
python -m waitress --port=5000 wsgi:app
```

#### Linux & Containerized Environments (Gunicorn):
```bash
pip install gunicorn
gunicorn --workers 1 --threads 1 --bind 0.0.0.0:5000 wsgi:app
```
> **Note**: Use `--threads 1` (single thread). ChromaDB 1.x uses a native Rust/Tokio
> runtime (`chromadb_rust_bindings`) that must be called from the thread on which it
> was initialized. Running with `--threads >1` causes Gunicorn's `gthread` pool to
> invoke ChromaDB from foreign threads, deadlocking `collection.query()`. PyTorch was
> replaced by ONNX Runtime (Phase 9D.1) so memory stays ~200MB, well within 512MB limits.

### Step 3.4: Verify Backend Health
```bash
curl http://127.0.0.1:5000/health
```
Expected response:
```json
{
  "status": "healthy",
  "service": "SchemeIQ+ Production API",
  "version": "1.0.0",
  "xgboost_model_available": true,
  "official_schemes_count": 14
}
```

---

## 4. Frontend Deployment

### Step 4.1: Configure API Base URL
In `frontend/.env` (or via CI/CD environment variables):
```ini
VITE_API_BASE_URL=https://api.schemeiq.example.com
```
*(For local production-like testing, use `http://127.0.0.1:5000`)*

### Step 4.2: Build Static Production Assets
```bash
cd frontend
npm install
npm run build
```
This generates optimized, minified static files in `frontend/dist/`:
- `dist/index.html`
- `dist/assets/*.js`
- `dist/assets/*.css`

### Step 4.3: Serve Frontend Static Assets

#### Option A — Local Production-like Smoke Testing:
```bash
npx serve -s dist -l 5173
```

#### Option B — Nginx Configuration:
```nginx
server {
    listen 80;
    server_name schemeiq.example.com;

    root /var/www/schemeiq/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 5. Security & Invariant Checklist

Before directing live citizen traffic to the deployment, verify the following invariants:

1. **Deterministic Eligibility Authority**:
   - The backend rule engine is the sole authority for scheme eligibility (`ELIGIBLE`, `POTENTIALLY_ELIGIBLE`, `NOT_ELIGIBLE`).
   - Disqualified schemes are never recommended.

2. **Advisory ML Safe Degradation**:
   - The XGBoost model acts purely as an advisory re-ranking layer over already-eligible candidates.
   - If the model is absent or throws an exception, the system seamlessly falls back to Phase 6 transparent heuristic ranking without throwing HTTP 500 errors.

3. **Strict Data Minimization & Privacy**:
   - The API pre-screens and rejects prohibited identifiers (`aadhaar_number`, `pan_number`, `bank_account_number`) with HTTP 400 `PRIVACY_VIOLATION`.

4. **Production Debug Mode Disabled**:
   - When `FLASK_ENV=production`, debug mode is forced to `False` to prevent stack trace leaks.

5. **CORS Restricted**:
   - In production, set `CORS_ALLOWED_ORIGINS` to the exact public frontend domain(s) instead of `*`.

---

## 6. Verification Commands

Run the full automated test suite to confirm operational readiness:

```bash
# 1. Full regression test suite (86+ tests)
python -m pytest -v

# 2. Statutory corpus and vector store integrity check
python src/eligibility/_check_integrity.py

# 3. End-to-end integration and smoke test
python scripts/e2e_smoke_test.py
```
