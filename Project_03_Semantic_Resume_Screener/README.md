# Semantic resume screener and talent search portal

A machine learning project combining a domain fine-tuned sentence transformer pipeline with an interactive Next.js 15 web portal.

## 1. Architecture overview

```text
Project_03_Semantic_Resume_Screener/
├── Resume.csv                 # 2,484 resumes across 24 categories
├── requirements.txt           # Python dependencies for the ML pipeline
├── ml_pipeline/               # Python ML training, evaluation, and CLI screener
│   ├── __init__.py
│   └── tests/                 # Pytest test suite for ML components
│       ├── test_pair_generation.py
│       ├── test_training.py
│       └── test_cli_screener.py
└── web_portal/                # Next.js 15 App Router, TypeScript, Tailwind CSS
    ├── app/                   # App Router pages and API routes
    ├── lib/                   # Extractor, PII scrubber, and scoring logic
    ├── package.json           # Node.js dependencies
    └── vitest.config.ts       # Vitest test configuration
```

### Machine learning pipeline (`ml_pipeline/`)
- Base model: `sentence-transformers/all-MiniLM-L6-v2` with 384-dimensional embeddings.
- Objective: Domain fine-tuning using MultipleNegativesRankingLoss (MNRL) on category-conditioned pairs with in-batch negatives.
- Evaluation: Validation loss curves, cosine separation distributions, and MRR metrics.
- Export: INT8 quantized ONNX model and a standalone CLI screener script (`cli_screener.py`).

### Web portal (`web_portal/`)
- Framework: Next.js 15 App Router, React 19, TypeScript, and Tailwind CSS.
- Browser extraction and PII scrubbing: Resumes parse in the browser using `pdfjs-dist` and undergo regex filtering (`[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, `[REDACTED_URL]`) before sending compact payloads to serverless routes.
- Scoring and embeddings: Pluggable provider architecture using Gemini Flash for structured parsing and Gemini `text-embedding-004` (or local ONNX) for vector generation, with instant client-side weight adjustments for skills, experience, and education.

## 2. Dataset configuration

The dataset `Resume.csv` contains 2,484 candidate resumes across 24 industry categories. To set up the data environment locally:

```bash
git lfs install
git lfs pull
```

## 3. Web portal quickstart

Prerequisites:
- Node.js v18.18+ (tested on Node v24)
- npm 10+

Setup and development:
```bash
cd web_portal
npm install
npm run dev
```

Run tests and production build:
```bash
npx vitest run
npm run build
```

## 4. ML pipeline quickstart

Prerequisites:
- Python 3.12+
- uv or standard pip / venv

Environment setup:
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Run test suite:
```bash
pytest ml_pipeline/tests
```

## 5. Project Architecture & Engineering Highlights

- **Contrastive Learning:** Fine-tunes `all-MiniLM-L6-v2` using Multiple Negatives Ranking Loss (MNRL) on category-paired job descriptions and resumes, mapping technical terminology into a shared 384-dimensional dense vector space.
- **Low-Latency Inference:** Quantized weights to 8-bit integers (INT8) and exported to ONNX Runtime (`model_quantized.onnx`), cutting model size to 58MB and reducing CPU inference latency to ~30ms per candidate.
- **Demographic Bias Mitigation:** Deterministic regex scrubbing runs client-side to redact candidate names, email addresses, phone numbers, and URLs before embedding or displaying profiles.
- **Explainable Multi-Factor Scoring:** Combines semantic embedding similarity, requirement recall coverage, experience tenure ratios, and degree levels with transparent skill gap auditing. Bonus candidate skills are surfaced without diluting match percentages.
