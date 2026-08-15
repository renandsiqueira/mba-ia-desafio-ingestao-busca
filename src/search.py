import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

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

def get_gemini_store():
    embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL","models/gemini-embedding-001"))
    store = PGVector(
        embeddings=embeddings,
        collection_name=f"{os.getenv('PG_VECTOR_COLLECTION')}-gemini",
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )
    return store

def get_openai_store():
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small"))
    store = PGVector(
        embeddings=embeddings,
        collection_name=f"{os.getenv('PG_VECTOR_COLLECTION')}-openai",
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )
    return store

def run_rag(store, question):
    results = store.similarity_search_with_score(question, k=10)
    
    # 1. Formatando os documentos, agora desempacotando a tupla (doc, score)
    formatted_docs = []
    for i, (doc, score) in enumerate(results, start=1):
        metadata_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
        doc_str = (
            f"--- Documento {i} (Score de similaridade: {score:.4f}) ---\n"
            f"Fonte/Metadados: {metadata_str}\n"
            f"Texto: {doc.page_content.strip()}"
        )
        formatted_docs.append(doc_str)
    
    context_str = "\n\n".join(formatted_docs)
    
    # 2. Preparando o LLM e a Chain
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Chain simples e direta do LCEL
    rag_chain = prompt | llm | StrOutputParser()
    
    # Retorna estritamente a string gerada pela IA
    return rag_chain.invoke({"contexto": context_str, "pergunta": question})

def search_prompt():
    for k in ("PDF_PATH","PG_VECTOR_COLLECTION", "DATABASE_URL"):
      if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")
        
    # Menus de interação sem print no retorno final da resposta
    print("Escolha o modelo de embeddings para busca vetorial:")
    print("1. Google Generative AI Embeddings (Gemini)")
    print("2. OpenAI Embeddings")
    choice = input("Digite 1 ou 2: ")

    if choice == "1":
        store = get_gemini_store()
    elif choice == "2":
        store = get_openai_store()
    else:
        print("Escolha inválida.")
        return

    question = input("Digite sua pergunta: ")
    
    # Imprime apenas o resultado final
    resposta = run_rag(store, question)
    print(f"\n{resposta}\n")
    return resposta

if __name__ == "__main__":
    search_prompt()