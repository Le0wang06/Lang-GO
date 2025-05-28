import os
from langchain_openai import OpenAIEmbeddings

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings()

# Create embedding
query_embedding = embeddings_model.embed_query("What is the meaning of life?")

print(query_embedding)