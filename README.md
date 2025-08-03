
# RAG Demo UI (Streamlit + LangChain)

A minimal web UI for your local RAG pipeline.

## Features
- Load docs from a folder (.txt, .md, .pdf, .docx)
- Chunk + embed (OpenAI embeddings)
- In-memory vector index (fast demo)
- Configurable retrieval (similarity / MMR)
- Optional company-aware filtering (based on `Company:` metadata)
- Answer viewer + context preview + sources

## Setup (Windows PowerShell)

```powershell
cd rag_ui_streamlit
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env and paste your OpenAI API key
streamlit run app.py
```

Then open the URL shown in your terminal (usually http://localhost:8501).

## Tips
- Put your documents in one folder and paste the path in the sidebar.
- Click **Build / Refresh Index** after adding or changing docs.
- Toggle **Context preview** to see what the model sees.
- Turn off company filtering for general questions (counts/lists).
- For persistence, replace `InMemoryVectorStore` with Chroma.
