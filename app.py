"""
A simple Streamlit app demonstrating Vectorless RAG using PageIndex.
Each line is commented for clarity.
API keys are loaded from config.py (not shown in UI).
"""

import streamlit as st  # Streamlit for web UI
import requests  # For downloading PDF
import os  # For file operations
from pageindex import PageIndexClient  # PageIndex SDK
import pageindex.utils as utils  # PageIndex utilities
import json  # For JSON operations
import openai  # For LLM calls
from config import OPENAI_API_KEY, PAGEINDEX_API_KEY  # Load API keys from config file

# --- Sidebar: Project Introduction ---
st.sidebar.title("Vectorless RAG Demo")
st.sidebar.markdown("""
**How is this different from traditional RAG?**
- **No Vectors:** Uses document structure and LLM reasoning, not vector DBs.
- **No Chunking:** Uses natural document sections, not artificial chunks.
- **Human-like Retrieval:** Simulates expert navigation of documents.
- **Transparent:** Retrieval is reasoning-based, not just top-k similarity.

**Traditional RAG:**
- Splits docs into chunks, embeds as vectors, retrieves by similarity.
- May miss context, less interpretable.

**Vectorless RAG (PageIndex):**
- Builds a tree index of the document.
- Uses LLM to reason over the tree for relevant sections.
- No vector search or chunking needed.
""")

# --- Main App ---
st.title("Vectorless RAG with PageIndex")
st.write("""
This demo shows how to use PageIndex for Retrieval-Augmented Generation (RAG) **without** vectors.
""")

# --- User Inputs ---
st.header("Step 1: Upload or Provide PDF")

# Create two columns for URL and file upload
col1, col2 = st.columns(2)  # Create two equal columns

with col1:
    st.subheader("Option 1: PDF URL")
    pdf_url = st.text_input("PDF URL", value="https://arxiv.org/pdf/2501.12948.pdf")  # Input for PDF URL

with col2:
    st.subheader("Option 2: Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")  # File uploader widget

if st.button("Run Vectorless RAG"):
    # --- Validate Inputs ---
    if not pdf_url and not uploaded_file:
        st.error("Please provide a PDF URL or upload a PDF file.")
    else:
        # --- Handle PDF Input (URL or File) ---
        if uploaded_file:
            # Use uploaded file directly
            st.write("Processing uploaded PDF...")
            os.makedirs("data", exist_ok=True)  # Create data folder if needed
            pdf_path = os.path.join("data", uploaded_file.name)  # Save with original filename
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())  # Save uploaded file
            st.success(f"Uploaded {uploaded_file.name}")
        else:
            # Download PDF from URL
            st.write("Downloading PDF...")
            pdf_path = os.path.join("data", pdf_url.split("/")[-1])  # Save in 'data' folder
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)  # Create folder if needed
            response = requests.get(pdf_url)  # Download PDF
            with open(pdf_path, "wb") as f:
                f.write(response.content)  # Save PDF
            st.success(f"Downloaded {pdf_url}")

        # --- Setup PageIndex Client ---
        pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)  # Auth with PageIndex

        # --- Submit Document to PageIndex ---
        st.write("Submitting document to PageIndex...")
        doc_id = pi_client.submit_document(pdf_path)["doc_id"]  # Upload PDF
        st.success(f"Document submitted. Doc ID: {doc_id}")

        # --- Wait for Indexing ---
        st.write("Waiting for PageIndex to process the document...")
        import time
        for _ in range(20):  # Wait up to ~20s
            if pi_client.is_retrieval_ready(doc_id):
                break
            time.sleep(1)
        else:
            st.error("PageIndex is still processing. Try again later.")
            st.stop()

        # --- Get PageIndex Tree ---
        tree = pi_client.get_tree(doc_id, node_summary=True)["result"]  # Get tree
        st.write("## Document Tree Structure (Simplified)")
        st.json(tree)  # Show tree structure

        # --- User Query ---
        st.header("Step 2: Ask a Question")
        query = st.text_input("Your Question", value="What are the conclusions in this document?")
        if st.button("Retrieve Answer"):
            # --- Prepare LLM Call ---
            openai.api_key = OPENAI_API_KEY  # Set OpenAI key from config
            tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])  # Remove text for prompt
            search_prompt = f"""
You are given a question and a tree structure of a document.\nEach node contains a node id, node title, and a summary.\nYour task is to find all nodes likely to contain the answer.\n\nQuestion: {query}\n\nDocument tree structure:\n{json.dumps(tree_without_text, indent=2)}\n\nReply in JSON:\n{{\n    \"thinking\": \"<Your reasoning>\",\n    \"node_list\": [\"node_id_1\", ...]\n}}\n"""
            # --- Call OpenAI LLM (sync for demo) ---
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use GPT-3.5 for demo (change as needed)
                messages=[{"role": "user", "content": search_prompt}],
                temperature=0
            )
            tree_search_result = response["choices"][0]["message"]["content"].strip()
            tree_search_result_json = json.loads(tree_search_result)
            st.write("### LLM Reasoning Process:")
            st.write(tree_search_result_json["thinking"])
            st.write("### Retrieved Node IDs:")
            st.write(tree_search_result_json["node_list"])

            # --- Map node IDs to content ---
            node_map = utils.create_node_mapping(tree)
            node_list = tree_search_result_json["node_list"]
            relevant_content = "\n\n".join(node_map[nid]["summary"] for nid in node_list if nid in node_map)
            st.write("### Retrieved Context:")
            st.write(relevant_content)

            # --- Generate Final Answer ---
            answer_prompt = f"""
Answer the question based on the context:\n\nQuestion: {query}\nContext: {relevant_content}\n\nProvide a clear, concise answer based only on the context provided.
"""
            response2 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=0
            )
            answer = response2["choices"][0]["message"]["content"].strip()
            st.write("## Final Answer:")
            st.success(answer)

# --- Footer ---
st.markdown("---")
st.markdown("""
**Note:** This is a minimal demo. For production, handle errors, rate limits, and use async for LLM calls.
""")