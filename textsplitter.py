from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sample text with multiple paragraphs
text = """
Artificial Intelligence (AI) is transforming our world in unprecedented ways. 
From healthcare to transportation, AI systems are making decisions that affect our daily lives.

Machine Learning, a subset of AI, enables computers to learn from data without being explicitly programmed. 
This technology powers everything from recommendation systems to autonomous vehicles.

Deep Learning, a more specialized form of Machine Learning, uses neural networks with many layers to process complex patterns. 
These networks can recognize images, understand speech, and even generate human-like text.

The impact of these technologies is profound. They're helping doctors diagnose diseases, 
enabling self-driving cars to navigate safely, and allowing computers to understand and generate human language.
"""

print("Original text length:", len(text), "characters")
print("\nSplitting text into chunks...\n")

# Initialize the splitter with smaller chunks for demonstration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # Smaller chunks to better see the splitting "how long each chunk should be"
    chunk_overlap=50,  # how many characters (or tokens) overlap between chunks (helps preserve context)
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # Split by paragraphs, then lines, then words
)

# Split the text
chunks = text_splitter.split_text(text)

# Output the chunks with their sizes
print(f"Text split into {len(chunks)} chunks:\n")
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} (length: {len(chunk)} characters):")
    print("-" * 50)
    print(chunk)
    print("-" * 50)
    print()
