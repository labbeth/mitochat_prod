<h1>
  <img src="assets/logo.png" alt="MitoChat" width="50" style="vertical-align: middle;"/>
  MitoChat: RAG + Agentic Streamlit Application
</h1>

**MitoChat** is a Retrieval-Augmented Generation (RAG) application designed to support genomic and clinical knowledge exploration using a GPU-accelerated backend.  

The stack includes:

- A **Streamlit web interface** (frontend)
- A **FastAPI RAG backend** powered by:
  - **vLLM** (GPU-accelerated text generation)
  - **FAISS** (dense retrieval)
  - **MiniLM** (embeddings)
  - **BGE-reranker-base** (cross-encoder reranking)
  - **Helsinki MarianMT** (FR ↔ EN translation)
  - **spaCy** (optional, for sentence-level highlighting)
- A modular codebase to later add STT/TTS microservices (Whisper, Kokoro, etc.).

---

## 📁 Repository Structure

```text
.
├── assets/                     
├── data/                       
│   ├── index/
│   ├── clinvar/
│   ├── genereviews/
│   └── mitocarta/
│
├── models/                     
│   ├── sentence-transformers/all-MiniLM-L6-v2/
│   ├── BAAI/bge-reranker-base/
│   ├── Helsinki-NLP/opus-mt-fr-en/
│   └── Helsinki-NLP/opus-mt-en-fr/
│
├── prompts/                    
│   ├── rewrite_en.yaml
│   └── rewrite_fr.yaml
│
├── scripts/
│   ├── build_corpus_and_index_prod.py
│   ├── rag_core.py
│   ├── fastapi_backend.py
│   ├── streamlit_app_frontend.py
│   ├── pdf_rendering.py
│   ├── utils.py
│   └── config.yaml
│
├── docker/
│   ├── backend.Dockerfile
│   ├── frontend.Dockerfile
│
├── requirements.backend.txt
├── requirements.frontend.txt
├── docker-compose.yml
└── README.md
```

---

## ⚙️ Features

### Backend (FastAPI + RAG)

- French query ➜ **FR→EN translation** ➜ RAG in English ➜ **EN→FR translation**
- **FAISS** dense retrieval
- **MiniLM** embeddings 
- **BGE-reranker-base** reranking
- **vLLM** LLM serving (GPU)
- Optional **spaCy** integration for:
  - sentence splitting,
  - highlighting the most relevant sentences inside retrieved chunks.

### Frontend (Streamlit)

- Chat UI
- Sends French queries, receives French answers

---

## 🛠️ 1. Local Development (without Docker)

### Create & activate a virtual env

```bash
python -m venv venv_clean

# Windows
venv_clean\Scripts\activate

# Linux / macOS
source venv_clean/bin/activate

```

### Install deps

```bash
pip install -r requirements.backend.txt
pip install -r requirements.frontend.txt
```

If you want to run vLLM locally outside Docker, you also need:

```bash
pip install vllm
```

---

## 🧱 2. Build FAISS Index

The backend expects a FAISS index and docstore.

```bash
python scripts/build_corpus_and_index_prod.py
```

This will:
- Load documents from data/clinvar, data/genereviews, data/mitocarta, etc.
- Encode them with MiniLM / sentence-transformers
- Build a FAISS index
- Store metadata in `data/index/` and associated files.

---

## 🚀 3. Run Backend & Frontend Locally

Run backend (FastAPI + uvicorn):

```bash
uvicorn scripts.fastapi_backend:app --host 0.0.0.0 --port 9000 --reload
```

Open the interactive docs: http://localhost:9000/docs

Run frontend (Streamlit):

```bash
streamlit run scripts/streamlit_app_frontend.py
```

Access the UI: http://localhost:8501

---

## 🐳 4. Docker Setup

This project uses **two Docker images**:
- `backend.Dockerfile`: FastAPI + vLLM + RAG (GPU)
- `frontend.Dockerfile`: Streamlit UI (CPU)


### 4.1 Backend Dockerfile (overview)

The backend image:

- Uses a **CUDA 12.1** runtime base image  
- Installs **Python**, **PyTorch (cu121)**, **vLLM**  
- Installs all backend dependencies from `requirements.backend.txt`  
- Copies backend code, scripts, and prompts  
- **Does NOT include models or data** → they are mounted as **volumes**  
- Exposes **port 9000**  
- Launches FastAPI via:

```bash
uvicorn scripts.fastapi_backend:app --host 0.0.0.0 --port 9000
```

This image is GPU-enabled and requires:

- Host NVIDIA drivers  
- NVIDIA Container Toolkit  
- `--gpus all` (compose) or device reservation  


### 4.2 Frontend Dockerfile (overview)

The frontend image:

- Uses **python:3.11-slim**  
- Installs dependencies via `requirements.frontend.txt`  
- Copies `scripts/` and `assets/`  
- Exposes **port 8501**  
- Runs Streamlit:

```bash
streamlit run scripts/streamlit_app_frontend.py --server.address=0.0.0.0
```

The frontend container has **no GPU requirements**.


### 4.3 Building Docker Images

From the project root:

### Backend:

```bash
docker build -f docker/backend.Dockerfile -t mitochat-backend .
```

### Frontend:

```bash
docker build -f docker/frontend.Dockerfile -t mitochat-frontend .
```

---

## 🧩 5. Docker Compose Deployment

```bash
docker compose up -d --build
```

This starts:

- **Backend** at port **9000**  
- **Frontend** at port **8501**  

Check logs:

```bash
docker compose logs -f backend
docker compose logs -f frontend
```

---

## 🧠 6. GPU / vLLM Notes

The backend image includes:

- PyTorch + CUDA 12.1  
- `vllm`  

Requirements on host:

- NVIDIA driver  
- NVIDIA Docker Toolkit  
- `docker run --gpus all ...`  

Check GPU access:

```bash
docker exec -it mitochat_backend python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 🌐 7. FastAPI Endpoints

Open API documentation: http://localhost:9000/docs

Example request:

```bash
curl -X POST http://localhost:9000/rag/query   -H "Content-Type: application/json"   -d '{"query": "Que sais-tu sur MT-ND1 ?"}'
```

---

## 🖥️ 8. Production Architecture (Example)

A target setup might look like:

```text
Internet Users
    |
    |  TCP 443 (HTTPS)
    v
   WAF
    |
    |  TCP 443 (HTTPS)
    v
[ DMZ SERVER ]
    ├─ Reverse Proxy (nginx/caddy)
    │      └─ forwards to Streamlit UI (localhost:8501)
    │
    └─ Streamlit Web App (Frontend UI)
           ├─ Displays chat interface
           ├─ Sends raw user query (FR) → RAG API
           └─ Receives final FR answer

[ GPU SERVER ]
    ├─ RAG Backend API (FastAPI + vLLM)
    │     ├─ Translation (FR↔EN)
    │     ├─ RAG core (FAISS + embeddings + reranker)
    │     └─ LLM generation with vLLM
    │
    ├─ Local assets:
    │     ├─ Models (translation, embeddings, reranker, LLM)
    │     └─ FAISS index & docstore (mounted under /app/data)

```

## 📄 License

MIT License.
