<div align="center">

# ⚙️ Fixora — AI-Powered Industrial Field Service Workspace

**A grounded AI maintenance copilot designed for on-site biomedical and industrial technicians.**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-LPU_Inference-f55036?style=for-the-badge)](https://groq.com/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)

</div>

---

## 📖 Overview

**Fixora** turns complex, multi-thousand-page technical service manuals into an instant, conversational, and voice-enabled diagnostic workspace. 

Instead of searching through dense PDF manuals under critical field conditions, technicians can describe a fault, error code, or symptom by voice or text. Fixora retrieves exact manual excerpts, enforces safety hazard checks, and delivers step-by-step procedures with exact page citations.

---

## ✨ Key Features

- **💬 ChatGPT-Style Industrial Workspace**: Clean modern conversational interface with multi-session chat history, dynamic suggestion pills, and dark/light themes.
- **🛡️ Strict Grounding & Anti-Hallucination**: Answers are derived strictly from indexed service manuals. Out-of-domain queries trigger a safe fallback.
- **⚠️ Safety Hazard Detection**: Automatically flags high-voltage, radiation, chemical, and pressure risks with high-priority warnings (e.g. LOTO protocols).
- **📋 Procedure Execution Cards**: Formats complex repairs into numbered, sequential checklists with step counters and source manual references.
- **🎙️ Hands-Free Voice Call Hub**: Browser-native speech recognition and speech synthesis powered by a minimal animated gradient orb.
- **⚡ Ultra-Low Latency Inference**: Powered by Groq LPU inference (`qwen/qwen3.8-27b`) with automated fallback routing.
- **📊 Automated Evaluation Benchmark**: Built-in 77-point test suite (`evaluate_rag.py`) measuring retrieval, faithfulness, safety, and output formatting.

---

## 🏗️ Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │   Technician Input (Chat UI / Voice Mic)     │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 1. Hybrid Retrieval Engine (ChromaDB)        │
                  │    - Semantic Embeddings (8,117 chunks)      │
                  │    - Exact Error Code Table Row Matching     │
                  │    - Device-filtered Search                  │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 2. Grounded Context Construction             │
                  │    - Top-K manual pages & sections assembly  │
                  │    - Safety & hazard guardrails injection    │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 3. Groq LLM (Senior Field Engineer Persona)  │
                  │    - Strict reasoning over manual context    │
                  │    - Structured JSON output schema           │
                  │    - Conversational speech synthesis text    │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 4. Validation & Safety Enforcement           │
                  │    - Critical hazard flag verification       │
                  │    - Error code validation against context   │
                  │    - Web Fallback if NOT_FOUND in manuals    │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 5. Dual Rendering (UI & Voice Synthesis)     │
                  │    - Procedure Execution Card (Steps 01, 02) │
                  │    - Compact Source Citations (Manual & Page)│
                  │    - Browser SpeechSynthesis for Voice Mode  │
                  └──────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/mennah29/Fixora.git
cd Fixora
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Add your **Groq API Key** in `.env`:
```ini
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

### 4. Build Vector Store (First Run)
```bash
python scripts/build_index.py
```

### 5. Launch the Application

In **Terminal 1** (FastAPI Backend):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In **Terminal 2** (Streamlit Frontend Workspace):
```bash
streamlit run app.py --server.port 8501
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🧪 Running Automated Evaluation

Run the automated 77-assertion benchmark suite against the live system:
```bash
python evaluate_rag.py
```

Outputs a full category scorecard across:
- **Retrieval Grounding**: Verified manual page hits
- **Answer Faithfulness**: Keyword matching and procedure accuracy
- **Safety Compliance**: High-voltage and hazard alarm detection
- **Format Integrity**: Clean markdown and structured procedure cards

---

## 📁 Repository Structure

```
fixora/
├── app/
│   ├── config.py             # Settings & environment configuration
│   ├── main.py               # FastAPI application & endpoints
│   ├── schemas.py            # Pydantic request/response models
│   └── service.py           # Core RAG retrieval & LLM pipeline
├── data/
│   └── all_device_fault_chunks.json   # Processed manual chunks
├── scripts/
│   └── build_index.py        # ChromaDB index builder
├── app.py                    # Streamlit ChatGPT-style workspace
├── evaluate_rag.py           # Automated evaluation suite
├── evaluation_report.json    # Benchmark scorecard audit
├── voice_component.html      # Hands-free voice call orb component
├── requirements.txt          # Production dependencies
├── Dockerfile                # Container definition
├── docker-compose.yml        # Multi-container deployment
└── README.md
```

---

## 📄 License
MIT License. Created by [Menna Ashraf](https://github.com/mennah29).
