# Vectorless RAG using PageIndex

This project demonstrates a simple Retrieval-Augmented Generation (RAG) workflow **without vectors** using [PageIndex](https://pageindex.ai/) and Streamlit.

## How is this different from traditional RAG?
- **No Vectors:** Uses document structure and LLM reasoning, not vector DBs.
- **No Chunking:** Uses natural document sections, not artificial chunks.
- **Human-like Retrieval:** Simulates expert navigation of documents.
- **Transparent:** Retrieval is reasoning-based, not just top-k similarity.

**Traditional RAG:**
- Splits docs into chunks, embeds as vectors, retrieves by similarity.
- May miss context, less interpretable.

**Vectorless RAG (PageIndex):**
- Builds a tree index of the document.
- No vector search or chunking needed.
- PageIndex creates a tree structure of the document, organizing it hierarchically by sections and subsections
- Each node in the tree has an ID, title, and summary
This tree is stored in session state for reuse

## How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up API keys:
   - Open `config.py`
   - Replace the placeholder values with your actual API keys:
     - Get OpenAI API key from [OpenAI](https://platform.openai.com/api-keys)
     - Get PageIndex API key from [PageIndex Dashboard](https://dash.pageindex.ai/api-keys)

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Enter a PDF URL and ask a question about the document!

## Configuration

API keys are stored in `config.py` and **not shown in the UI**. The `.gitignore` file ensures `config.py` is never committed to version control to protect your API keys.

## Notes
- This is a minimal demo. For production, handle errors, rate limits, and use async for LLM calls.
- Do not share your `config.py` file or commit it to version control.
