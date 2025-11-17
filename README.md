# ![MitoChat](assets/logo.png)🧬 MitoChat — RAG + Agentic Streamlit Application

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
- Optional **spaCy** sentence highlighting

### Frontend (Streamlit)

- Chat UI
- Sends French queries, receives French answers

---

## 🛠️ 1. Local Development

### Create virtual env

```bash
python -m venv venv_clean
source venv_clean/bin/activate
```

### Install deps

```bash
pip install -r requirements.backend.txt
pip install -r requirements.frontend.txt
```

Run backend:

```bash
uvicorn scripts.fastapi_backend:app --reload --port 9000
```

Run frontend:

```bash
streamlit run scripts/streamlit_app_frontend.py
```

---

## 🧱 2. Build FAISS Index

```bash
python scripts/build_corpus_and_index_prod.py
```

Index is saved under `data/index/`.

---

## 🐳 3. Docker Setup

### Build backend:

```bash
docker build -f docker/backend.Dockerfile -t mitochat-backend .
```

### Build frontend:

```bash
docker build -f docker/frontend.Dockerfile -t mitochat-frontend .
```

---

## 🧩 4. Docker Compose Deployment

```bash
docker compose up -d --build
```

---

## 🧠 GPU Notes

- NVIDIA drivers required
- NVIDIA Docker Toolkit required
- vLLM is installed **inside** the backend container

Test GPU inside container:

```bash
docker exec -it mitochat_backend python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 🌐 FastAPI Endpoints

Visit:

```
http://localhost:9000/docs
```

---

## 📄 License

MIT License.
