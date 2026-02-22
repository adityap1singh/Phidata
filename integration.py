import pandas as pd
import phi
from pinecone import Vector
from pypdf import PdfReader
from phi.document import Document

def load_csv(file_path):
    df = pd.read_csv(file_path)
    documents = []

    for _, row in df.iterrows():
        content = f"""
        Ticket ID: {row.get('ticket_id')}
        Customer Message: {row.get('message')}
        Timestamp: {row.get('timestamp')}
        """
        documents.append(Document(text=content))
    return documents


def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [Document(text=text)]


def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = "\n".join([p.extract_text() for p in reader.pages])
    return [Document(text=text)]

from phi.document import Document
# from phi.chunking import RecursiveChunker

# If chunking functionality is needed, implement a simple chunker here or install the required package.
# For now, the chunk_documents function will just return the documents as-is.

def chunk_documents(documents):
    # Placeholder: returns documents without chunking
    return documents


2. 
import os
# from ingestion.csv_loader import load_csv_tickets
# from ingestion.txt_loader import load_txt_logs
# from ingestion.pdf_loader import load_pdf_docs

# Use the local loader functions defined above
load_csv_tickets = load_csv
load_txt_logs = load_txt
load_pdf_docs = load_pdf

SUPPORTED_EXTENSIONS = [".csv", ".txt", ".pdf"]

def handle_upload(file_path: str):
    _, ext = os.path.splitext(file_path.lower())

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".csv":
        return load_csv_tickets(file_path)

    if ext == ".txt":
        return load_txt_logs(file_path)

    if ext == ".pdf":
        return load_pdf_docs(file_path)
    
    
    
 
    import pandas as pd
from phi.document import Document

def load_csv_tickets(file_path: str):
    df = pd.read_csv(file_path)

    documents = []

    for _, row in df.iterrows():
        content = f"""
Customer Issue:
{row.get('issue', '')}

Timestamp:
{row.get('timestamp', '')}

Sentiment (if tagged):
{row.get('sentiment', 'Not Provided')}

Previous Response:
{row.get('response', 'Not Provided')}
""".strip()

        documents.append(
            Document(
                text=content,
                metadata={
                    "source": "csv",
                    "timestamp": row.get("timestamp"),
                    "sentiment": row.get("sentiment")
                }
            )
        )

    return documents



from phi.document import Document

def load_txt_logs(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [
        Document(
            text=text,
            metadata={
                "source": "txt"
            }
        )
    ]
    
    
    
    import pdfplumber
from phi.document import Document

def load_pdf_docs(file_path: str):
    text_chunks = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)

    full_text = "\n".join(text_chunks)

    return [
        Document(
            text=full_text,
            metadata={
                "source": "pdf",
                "document_type": "policy"
            }
        )
    ]
    
    
    
    
    import fitz  # PyMuPDF
from phi.document import Document

def load_pdf_docs_fitz(file_path: str):
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return [
        Document(
            text=text,
            metadata={
                "source": "pdf",
                "document_type": "policy"
            }
        )
    ]
    List[phi.document.Document]
    
  
  
  
from phi.document import Document

# Uncomment the following import if phi.chunking is available
# from phi.chunking import RecursiveChunker

# If RecursiveChunker is not available, comment out or remove the following chunker initializations.
# For support conversations (CSV / TXT)
# conversation_chunker = RecursiveChunker(
#     chunk_size=400,
#     chunk_overlap=80
# )

# For policy documents (PDF)
# policy_chunker = RecursiveChunker(
#     chunk_size=900,
#     chunk_overlap=120
# )




def chunk_conversations(documents: list[Document]) -> list[Document]:
    """
    Chunk customer conversations and internal logs (no-op placeholder)
    """
    return documents

def chunk_policies(documents: list[Document]) -> list[Document]:
    """
    Chunk policy and documentation PDFs (no-op placeholder)
    """
    return documents

from sentence_transformers import SentenceTransformer
from phi.document import Document
import numpy as np

model = SentenceTransformer("all-mpnet-base-v2")

def embed_documents(documents: list[Document]):
    texts = [doc.text for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)

    for doc, vector in zip(documents, embeddings):
        doc.embedding = np.array(vector)

    return documents

import google.generativeai as genai
from phi.document import Document

genai.configure(api_key="AIzaSyASYB6Pas5MCBUpZmsoepkbPjD1e9d76pg")

def embed_documents_gemini(documents: list[Document]):
    for doc in documents:
        response = genai.embed_content(
            model="models/embedding-001",
            content=doc.text
        )
        doc.embedding = response["embedding"]

    return documents


# Use the local embed_documents function defined above

def process_documents(documents):
    conversation_docs = [d for d in documents if d.metadata.get("source") != "pdf"]
    policy_docs = [d for d in documents if d.metadata.get("source") == "pdf"]

    chunked_docs = []
    chunked_docs.extend(chunk_conversations(conversation_docs))
    chunked_docs.extend(chunk_policies(policy_docs))

    embedded_docs = embed_documents(chunked_docs)
    return embedded_docs

Document(
  text="chunk content",
  metadata={
    "source": "csv/pdf/txt",
    "timestamp": "...",
    "sentiment": "..."
  },
  embedding=[0.021, -0.44, ...]
)



import pinecone
from phi.document import Document

pinecone.init(
api_key="YOUR_PINECONE_API_KEY",
environment="us-east-1"
)

INDEX_NAME = "support-triage-index"

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=768, # sentence-transformer mpnet
        metric="cosine"
    )
index = pinecone.Index(INDEX_NAME)

def upsert_documents(documents: list[Document]):
vectors = []

for i, doc in enumerate(documents):
vectors.append((
f"doc-{i}",
doc.embedding,
doc.metadata
))

index.upsert(vectors=vectors)

def semantic_search(query, filters=None, top_k=5):
    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filters
    )
    return results

from embeddings.sentence_transformer import model

def semantic_search(query, filters=None, top_k=5):
    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filters
    )
    return results

filters = {
    "sentiment": "Negative",
    "timestamp": {"$gte": "2024-05-01", "$lte": "2024-05-31"}
}

from textblob import TextBlob
from datetime import datetime

def sentiment_tool(text: str):
    polarity = TextBlob(text).sentiment.polarity
    return "Negative" if polarity < -0.3 else "Positive" if polarity > 0.3 else "Neutral"


def urgency_tool(text: str):
    keywords = ["urgent", "immediately", "asap", "now", "critical"]
    return "High" if any(k in text.lower() for k in keywords) else "Normal"


def intent_classifier(text: str):
    intents = {
        "refund": ["refund", "money back"],
        "delivery": ["late", "delay", "shipping"],
        "account": ["login", "password", "account"]
    }

    for intent, keys in intents.items():
        if any(k in text.lower() for k in keys):
            return intent
    return "general"
def policy_lookup(agent, query):
    return agent.run(
        f"Retrieve relevant policy sections related to: {query}"
    )
    
    def refund_eligibility(order_date: str, today: str, policy_days=30):
    order_dt = datetime.fromisoformat(order_date)
    today_dt = datetime.fromisoformat(today)

    delta = (today_dt - order_dt).days
    return delta <= policy_days
def external_lookup(query: str):
    return f"External lookup placeholder for: {query}"

from ingestion.pipeline import process_documents
from vectordb.pinecone_store import upsert_documents
from agent.tools import sentiment_tool, urgency_tool, intent_classifier

def triage_workflow(raw_documents):
    embedded_docs = process_documents(raw_documents)
    upsert_documents(embedded_docs)
    return "Documents successfully processed & indexed"

def analyze_ticket(agent, ticket_text):
    return {
        "sentiment": sentiment_tool(ticket_text),
        "urgency": urgency_tool(ticket_text),
        "intent": intent_classifier(ticket_text),
        "suggested_response": agent.run(
            f"Generate a professional support response for:\n{ticket_text}"
        )
    }
    import streamlit as st
from ingestion.upload_handler import handle_upload
from workflow.triage_workflow import triage_workflow
from agent.support_agent import SupportTriageAgent

st.set_page_config(layout="wide")
st.title("AI Customer Support Triage Agent")

uploaded_file = st.file_uploader(
    "Upload Support Logs or Policies",
    type=["csv", "txt", "pdf"]
)

if uploaded_file:
    path = f"data/uploads/{uploaded_file.name}"
    with open(path, "wb") as f:
        f.write(uploaded_file.read())

    docs = handle_upload(path)
    st.success(triage_workflow(docs))

st.subheader("Chat with Support Data")
query = st.text_input("Ask a question")

if query:
    response = agent.run(query)
    st.write(response)

st.subheader("Response Suggestion")
ticket = st.text_area("Paste a support ticket")

if st.button("Generate Draft Reply"):
    st.write(agent.run(f"Suggest a response for:\n{ticket}"))