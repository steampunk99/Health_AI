from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

PINECONE_API_KEY="8a429e99-a42c-4f0b-96ef-fc2097aa38c4"
PINECONE_API_ENV="gcp-starter"

# extract data from the pdf
def load_pdf(data):
    loader = DirectoryLoader(data,glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents

extracted_data = load_pdf("data/")