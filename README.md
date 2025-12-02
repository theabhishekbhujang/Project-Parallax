# Project Parallax

*See beyond the text, discover new insights.*

---

## Overview

Project Parallax is an AI-powered intelligent document assistant built with Streamlit and advanced language models. It allows you to upload PDF documents, processes them asynchronously, and enables you to ask questions about their content. Using semantic search and embeddings, Project Parallax delivers precise, context-aware answers by analyzing the most relevant parts of your documents.

---

## Features

- Upload PDF files for analysis  
- Asynchronous processing for smooth UI experience  
- Smart document chunking for focused context retrieval  
- Semantic search to find relevant document sections  
- AI-generated answers based on document content  
- "Think Process" insights displayed dynamically during response generation  
- Clean, dark-themed UI with intuitive interaction  

---

## Installation

Make sure you have Python 3.8+ installed.

Install dependencies:

```bash
pip install streamlit aiofiles langchain langchain_community langchain_text_splitters langchain_core langchain_ollama
```

## Usage
Run the Streamlit app:

```
streamlit run Project Parallax.py
```

- Then open the browser at the local address shown (usually http://localhost:8501).

- Upload a PDF file using the uploader.

- Once processed, ask questions in the input box.

- The assistant will provide answers based on the document content.
