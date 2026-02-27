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
from openai import OpenAI  # For LLM calls
from config import OPENAI_API_KEY, PAGEINDEX_API_KEY  # Load API keys from config file

# --- Initialize OpenAI Client ---
client = OpenAI(api_key=OPENAI_API_KEY)  # Initialize OpenAI client with API key

# --- Page Configuration ---
st.set_page_config(layout="wide")  # Set wide layout for full screen

# --- Initialize Session State ---
if 'processed_document' not in st.session_state:
    st.session_state.processed_document = False
if 'tree' not in st.session_state:
    st.session_state.tree = None
if 'doc_id' not in st.session_state:
    st.session_state.doc_id = None
if 'document_name' not in st.session_state:
    st.session_state.document_name = None

# --- Main App ---
st.title("Vectorless RAG with PageIndex")

# Create two columns
left_col, right_col = st.columns(2)

# --- COLUMN 1: Upload Document ---
with left_col:
    
    uploaded_file = st.file_uploader("", type="pdf")  # File uploader widget

    if st.button("Process Document"):
        # --- Validate Inputs ---
        if not uploaded_file:
            st.error("Please upload a PDF file.")
        else:
            # --- Handle PDF Input ---
            # Use uploaded file directly
            st.write("Processing uploaded PDF...")
            os.makedirs("data", exist_ok=True)  # Create data folder if needed
            pdf_path = os.path.join("data", uploaded_file.name)  # Save with original filename
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())  # Save uploaded file
            st.success(f"✅ Uploaded {uploaded_file.name}")

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
            
            # Store in session state
            st.session_state.processed_document = True
            st.session_state.tree = tree
            st.session_state.doc_id = doc_id
            st.session_state.document_name = uploaded_file.name
            st.rerun()

    # Display document summary if processed
    if st.session_state.processed_document:
        st.success(f"✅ Document Ready: {st.session_state.document_name}")
        st.write("### 📊 Document Index")
        with st.expander("View Document Tree Structure", expanded=True):
            st.json(st.session_state.tree)  # Show tree structure


# --- COLUMN 2: Ask Questions ---
with right_col:
    st.header("Ask Questions")
    
    if st.session_state.processed_document:
        query = st.text_input("Your Question", value="What are the main topics in this document?")
        
        if st.button("Get Answer"):
            # --- Prepare LLM Call ---
            tree_without_text = utils.remove_fields(st.session_state.tree.copy(), fields=['text'])  # Remove text for prompt
            search_prompt = f"""
You are given a question and a tree structure of a document.\nEach node contains a node id, node title, and a summary.\nYour task is to find all nodes likely to contain the answer.\n\nQuestion: {query}\n\nDocument tree structure:\n{json.dumps(tree_without_text, indent=2)}\n\nReply in JSON:\n{{\n    \"thinking\": \"<Your reasoning>\",\n    \"node_list\": [\"node_id_1\", ...]\n}}\n"""
            
            with st.spinner("Retrieving and generating answer..."):
                # --- Call OpenAI LLM (sync for demo) ---
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Use GPT-3.5 for demo (change as needed)
                    messages=[{"role": "user", "content": search_prompt}],
                    temperature=0
                )
                tree_search_result = response.choices[0].message.content.strip()
                tree_search_result_json = json.loads(tree_search_result)
                
                st.write("### 🧠 LLM Reasoning:")
                st.write(tree_search_result_json["thinking"])
                
                st.write("### 📍 Retrieved Nodes:")
                st.write(tree_search_result_json["node_list"])

                # --- Map node IDs to content ---
                node_map = utils.create_node_mapping(st.session_state.tree)
                node_list = tree_search_result_json["node_list"]
                relevant_content = "\n\n".join(node_map[nid]["summary"] for nid in node_list if nid in node_map)
                
                st.write("### 📚 Context:")
                st.write(relevant_content)

                # --- Generate Final Answer ---
                answer_prompt = f"""
Answer the question based on the context:\n\nQuestion: {query}\nContext: {relevant_content}\n\nProvide a clear, concise answer based only on the context provided.
"""
                response2 = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": answer_prompt}],
                    temperature=0
                )
                answer = response2.choices[0].message.content.strip()
                st.write("## ✨ Answer:")
                st.success(answer)
    else:
        st.info("👈 Upload and process a document on the left to start asking questions.")

# --- Footer ---
st.markdown("---")
st.markdown("""
**Note:** This is a minimal demo. For production, handle errors, rate limits, and use async for LLM calls.
""")