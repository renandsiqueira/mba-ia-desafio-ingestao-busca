import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def ingest_pdf():
    for k in ("PDF_PATH","PG_VECTOR_COLLECTION", "DATABASE_URL"):
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

def format_docs_with_metadata(docs):
    formatted_docs = []
    for i, doc in enumerate(docs, start=1):
        metadata_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
        doc_str = (
            f"--- Documento {i} ---\n"
            f"Fonte/Metadados: {metadata_str}\n"
            f"Texto: {doc.page_content.strip()}"
        )
        formatted_docs.append(doc_str)
    
    return "\n\n".join(formatted_docs)


def get_gemini_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL","models/gemini-embedding-001"))
    store = PGVector(
        embeddings=embeddings,
        collection_name=f"{os.getenv('PG_VECTOR_COLLECTION')}-gemini",
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )
    return store.as_retriever(search_kwargs={"k": 10})

def get_openai_retriever():
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small"))
    store = PGVector(
        embeddings=embeddings,
        collection_name=f"{os.getenv('PG_VECTOR_COLLECTION')}-openai",
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )
    return store.as_retriever(search_kwargs={"k": 10})


def run_rag_chain(retriever, question):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    
    rag_chain = (
        {"contexto": retriever | format_docs_with_metadata, "pergunta": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Retorna estritamente a string gerada pela IA
    return rag_chain.invoke(question)


def search_prompt():
    print("Escolha o modelo de embeddings para busca vetorial:")
    print("1. Google Generative AI Embeddings (Gemini)")
    print("2. OpenAI Embeddings")
    choice = input("Digite 1 ou 2: ")

    if choice == "1":
        retriever = get_gemini_retriever()
    elif choice == "2":
        retriever = get_openai_retriever()
    else:
        print("Escolha inválida.")
        return

    question = input("Digite sua pergunta: ")
    
    # Imprime apenas o resultado final
    resposta = run_rag_chain(retriever, question)
    print(f"\n{resposta}\n")


if __name__ == "__main__":
    search_prompt()