# YouTube Tutor — Retrieval-Augmented Generation

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square) ![FAISS](https://img.shields.io/badge/FAISS-087F5B?style=flat-square)

A learning project that converts available YouTube transcripts into a searchable knowledge base and answers questions using retrieved context and Google Gemini.

## Pipeline

YouTube URL → Transcript → Text chunks → Hugging Face embeddings → FAISS vector index → Context retrieval → Gemini response

## Setup

```bash
git clone https://github.com/DEEPAKRV07/RAG_LEARNIINGS.git
cd RAG_LEARNIINGS
python -m venv .venv
pip install -r requirements.txt
```

Activate your virtual environment before installing dependencies. Create a local `.env` file with:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

Never commit API keys or private data.

```bash
streamlit run app.py
```

Enter a YouTube video URL and ask a question after the application builds the index.

## Scope

This is an experimental RAG application, not a production service. It depends on transcript availability, model/API access and retrieval quality. Generated responses can be incomplete or inaccurate; verify against the original transcript. Local artifacts and logs are stored under `rag_data/`. Use content you have permission to process.
