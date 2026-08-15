import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

def ingest_pdf():
  for k in ("PDF_PATH","PG_VECTOR_COLLECTION"):
    if not os.getenv(k):
      raise RuntimeError(f"Environment variable {k} is not set")
    
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt_gemini(question=None):
    embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL","models/gemini-embedding-001"))
    store = PGVector(
        embeddings=embeddings,
        collection_name=f"{os.getenv('PG_VECTOR_COLLECTION')}-gemini",
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    results = store.similarity_search_with_score(question, k=10)

    for i, (doc, score) in enumerate(results, start=1):
      print("="*50)
      print(f"Resultado {i} (score: {score:.2f}):")
      print("="*50)

      print("\nTexto:\n")
      print(doc.page_content.strip())

      print("\nMetadados:\n")
      for k, v in doc.metadata.items():
        print(f"{k}: {v}")

def search_prompt_openai(question=None):
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small"))
    store = PGVector(
        embeddings=embeddings,
        collection_name=f"{os.getenv('PG_VECTOR_COLLECTION')}-openai",
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    results = store.similarity_search_with_score(question, k=10)

    for i, (doc, score) in enumerate(results, start=1):
      print("="*50)
      print(f"Resultado {i} (score: {score:.2f}):")
      print("="*50)

      print("\nTexto:\n")
      print(doc.page_content.strip())

      print("\nMetadados:\n")
      for k, v in doc.metadata.items():
        print(f"{k}: {v}")

def search_prompt():
    print("Escolha o modelo de embeddings para busca:")
    print("1. Google Generative AI Embeddings (Gemini)")
    print("2. OpenAI Embeddings")
    choice = input("Digite 1 ou 2: ")

    testQuestion = "Qual o faturamento da Empresa SuperTechIABrazil?"
    if choice == "1":
      question = input("Digite sua pergunta para busca com Google Generative AI Embeddings: ")
      search_prompt_gemini(testQuestion)
    elif choice == "2":
      question = input("Digite sua pergunta para busca com OpenAI Embeddings: ")
      search_prompt_openai(testQuestion)
    else:
      print("Escolha inválida. Por favor, digite 1 ou 2.")
      return None


if __name__ == "__main__":
    search_prompt()