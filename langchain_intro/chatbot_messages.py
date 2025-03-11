import dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

REVIEWS_CHROMA_PATH = "chroma_data/"

dotenv.load_dotenv()

# Use Hugging Face embeddings (MiniLM is small & efficient)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create the vector database
reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=embeddings,
)

# Query the vector database for relevant documents
question = """Has anyone complained about
           communication with the hospital staff?"""
relevant_docs = reviews_vector_db.similarity_search(question, k=3)

# Display the content of the most relevant documents
print(relevant_docs[0].page_content)
print(relevant_docs[1].page_content)
print(relevant_docs[2].page_content)
