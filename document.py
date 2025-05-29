import os
import getpass
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    
# print("\n1. Loading document...")
# # Load the document
# loader = TextLoader("state_of_the_union.txt")
# document = loader.load()
# print(f"Document loaded. Length: {len(document[0].page_content)} characters")

# print("\n2. Splitting document into chunks...")
# # Split the document into chunks
# text_splitter = CharacterTextSplitter(
#     separator="\n\n",
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len
# )
# texts = text_splitter.split_text(document[0].page_content)
# print(f"Document split into {len(texts)} chunks")
# print("\nFirst chunk preview:")
# print(texts[0][:200] + "...")

# print("\n3. Creating embeddings...")
# # Create embeddings and store them
# embeddings = OpenAIEmbeddings()
# db = FAISS.from_texts(texts, embeddings)
# print("Embeddings created successfully")

# print("\n4. Saving vector store...")
# # Save the vector store
# db.save_local("faiss_index")
# print("Vector store saved to 'faiss_index'")

# print("\n5. Testing similarity search...")
# # Test the vector store with a query
# query = "What are the main topics discussed?"
# results = db.similarity_search(query, k=2)
# print("\nMost relevant chunks for query:", query)
# for i, doc in enumerate(results, 1):
#     print(f"\nChunk {i}:")
#     print(doc.page_content[:200] + "...")
from langchain_core.vectorstores import InMemoryVectorStore

# Initialize with an embedding model
vector_store = InMemoryVectorStore(embedding=OpenAIEmbeddings())


document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

documents = [document_1, document_2]

vector_store.add_documents(documents=documents)

print(vector_store.similarity_search("What is the weather forecast for tomorrow?", k=1))