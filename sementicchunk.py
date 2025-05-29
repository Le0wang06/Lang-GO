from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

# ⚠️ Make sure your OpenAI API key is set
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Step 1: Define your input text
text = """LangChain is an open-source framework for building applications powered by large language models. 
It provides tools to connect language models to external data sources, interact with environments, 
and build advanced reasoning applications. LangChain supports multiple backends including OpenAI, HuggingFace, and more."""

# Step 2: Initialize the embedding model
embedding_model = OpenAIEmbeddings()

# Step 3: Use SemanticChunker to split the document
splitter = SemanticChunker(embedding_model)

# Wrap text into Document (required input type)
docs = [Document(page_content=text)]

# Step 4: Split the text
chunks = splitter.split_documents(docs)

# Step 5: Print the chunks
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content)
