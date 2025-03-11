import dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

REVIEWS_CSV_PATH = "./data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

reviews_vector_db = Chroma.from_documents(reviews, MistralAIEmbeddings(), persist_directory=REVIEWS_CHROMA_PATH)

