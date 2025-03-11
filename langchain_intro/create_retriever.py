import os
import dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

dotenv.load_dotenv()

# Load Hugging Face token
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is not set in the environment variables.")

os.environ["HF_TOKEN"] = hf_token

REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

# Load reviews from CSV
loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

# Load an embedding model from Hugging Face
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Create vector store and persist it
reviews_vector_db = Chroma.from_documents(
    documents=reviews,
    embedding=embeddings,
    persist_directory=REVIEWS_CHROMA_PATH
)

print("Vector store created and persisted successfully!")
