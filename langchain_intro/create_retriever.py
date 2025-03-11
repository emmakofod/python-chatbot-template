import dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

REVIEWS_CSV_PATH = "../data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

# Load the reviews from CSV
loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

# Use a Hugging Face model for embeddings (MiniLM is small & efficient)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create the vector database
reviews_vector_db = Chroma.from_documents(
    reviews, embeddings, persist_directory=REVIEWS_CHROMA_PATH
)

print("Vector DB created successfully!")
