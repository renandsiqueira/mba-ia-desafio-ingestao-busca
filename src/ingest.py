import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def ingest_pdf():
  for k in ("PDF_PATH","PG_VECTOR_COLLECTION"):
    if not os.getenv(k):
      raise RuntimeError(f"Environment variable {k} is not set")

  PDF_PATH = os.getenv("PDF_PATH")

  docs = PyPDFLoader(str(PDF_PATH)).load()

  splits = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=150, add_start_index=False).split_documents(docs)

  if not splits:
    raise SystemExit(0)

  enriched = [
    Document(
      page_content=d.page_content,
      metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
    )
    for d in splits
  ]    

  ingest_pdf_google_gen_ai(enriched)
  ingest_pdf_open_ai(enriched)

def ingest_pdf_open_ai(enriched):
  print("Iniciando ingestão com OpenAI Embeddings...")
  embeddings = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
  )
  ids = [f"doc-openai-{i}" for i in range(len(enriched))]
  collection_name=f"{os.getenv('PG_VECTOR_COLLECTION')}-openai"

  # Sem rate limit: batch_size=None e sleep_time=0
  store_pdf_in_pgvector(
    embeddings=embeddings,
    enriched=enriched,
    ids=ids,
    collection_name=collection_name,
    batch_size=None,
    sleep_time=0
  )
  print("Ingestão com OpenAI Embeddings concluída com sucesso!")

def ingest_pdf_google_gen_ai(enriched):
  print("Iniciando ingestão com Google Generative AI Embeddings...")
  embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
  )
  ids = [f"doc-gemini-{i}" for i in range(len(enriched))]
  collection_name=f"{os.getenv('PG_VECTOR_COLLECTION')}-gemini"

  # Com rate limit: envia 5 documentos por vez e aguarda 10 segundos
  store_pdf_in_pgvector(
    embeddings=embeddings,
    enriched=enriched,
    collection_name=collection_name,
    ids=ids,
    batch_size=5,
    sleep_time=10
  )
  print("Ingestão com Google Generative AI Embeddings concluída com sucesso!")

def store_pdf_in_pgvector(embeddings, collection_name, enriched, ids, batch_size=None, sleep_time=0):
  store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=os.getenv("DATABASE_URL"),
    use_jsonb=True,
  )

  print(f"Total de fragmentos para ingestão: {len(enriched)}")

  # Lógica para aplicar o rate limit (Batching)
  if batch_size and batch_size > 0:
    print("Iniciando ingestão COM rate limit...")
    for i in range(0, len(enriched), batch_size):
      batch_docs = enriched[i : i + batch_size]
      batch_ids = ids[i : i + batch_size]

      print(f"  -> Ingerindo lote {i//batch_size + 1}... (Itens {i} a {i + len(batch_docs) - 1})")

      store.add_documents(documents=batch_docs, ids=batch_ids)

      # Aplica o delay entre lotes, exceto no último
      if sleep_time > 0 and (i + batch_size) < len(enriched):
        time.sleep(sleep_time)

    print("Ingestão em lotes concluída com sucesso!")

  # Lógica SEM rate limit (tudo de uma vez)
  else:
    print("Iniciando ingestão SEM rate limit (Bulk insert)...")
    store.add_documents(documents=enriched, ids=ids)
    print("Ingestão concluída com sucesso!")


if __name__ == "__main__":
    ingest_pdf()