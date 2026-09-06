# SchemeIQ+: A Multilingual Explainable AI Assistant for Personalized Government Scheme Discovery

SchemeIQ+ is a research-grade, explainable AI assistant designed to empower citizens in discovering and understanding applicable government welfare schemes. Using Retrieval-Augmented Generation (RAG) coupled with deterministic, verifiable eligibility reasoning, SchemeIQ+ bridges the awareness gap while guaranteeing that official government sources remain the inviolable ground truth.

---

## 🎯 Research Project Scope

- **Geographical Scope**: Telangana, India.
- **Scheme Coverage**:
  - Telangana State Government Schemes (e.g., Rythu Bandhu / Rythu Bharosa, Aasara Pensions, Kalyana Lakshmi / Shaadi Mubarak, Mahalakshmi Scheme, Gruha Jyothi, etc.).
  - Central Government Schemes applicable in Telangana (e.g., PM-KISAN, Ayushman Bharat - PMJAY, PM Awas Yojana, PMMVY, etc.).
- **Languages**: English (`en`) and Telugu (`te`).
- **Core Tenets**:
  - **Ground-Truth Rigor**: Only verified official Government Orders (G.O.s), policy guidelines, and notifications from official government portals (`telangana.gov.in`, `myscheme.gov.in`, official department portals) serve as knowledge bases.
  - **Zero Rule Hallucination**: Large Language Models never fabricate eligibility rules. Eligibility verification is executed deterministically against verified criteria.
  - **Explainability**: Clear, transparent rationales are generated for why a user is eligible or ineligible, alongside explicit prerequisite documentation requirements.

---

## 📂 Project Architecture

```
SchemeIQ-Plus/
├── data/
│   ├── schemes.csv        # Structured schema index of official schemes
│   └── documents/          # Authentic government policy documents & G.O.s (PDF/Text)
├── src/
│   ├── ingestion/          # Document loading, parsing, cleaning, and chunking pipelines
│   ├── retrieval/          # Cross-lingual vector indexing, semantic and hybrid retrieval
│   ├── eligibility/        # Deterministic criteria validation engine based on official rules
│   └── recommendation/     # Profile-based ranking and explainable rationale generation
├── tests/                  # Automated unit, integration, and validation tests
├── .env                    # Environment configuration & API credentials
├── .gitignore              # Git exclusion configuration
├── requirements.txt        # Foundational Python dependencies
└── README.md               # Project overview and documentation
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.11+ (Tested on Python 3.13)

### 2. Environment Setup
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Upgrade pip and install foundational dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verification
```powershell
pytest
```
