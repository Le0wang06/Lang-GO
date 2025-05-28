import os
from langchain_openai import OpenAIEmbeddings
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings()

# Create embeddings for different queries
query1 = "What is the meaning of life?"
query2 = "What is the purpose of existence?"
query3 = "What is the weather like today?"

# Get embeddings
embedding1 = embeddings_model.embed_query(query1)
embedding2 = embeddings_model.embed_query(query2)
embedding3 = embeddings_model.embed_query(query3)

# Calculate similarities
similarity_same = cosine_similarity(embedding1, embedding1)  # Should be 1.0
similarity_similar = cosine_similarity(embedding1, embedding2)  # Should be high
similarity_different = cosine_similarity(embedding1, embedding3)  # Should be lower

print(f"\nSimilarity between same query: {similarity_same:.4f}")
print(f"Similarity between similar queries: {similarity_similar:.4f}")
print(f"Similarity between different queries: {similarity_different:.4f}")