# Desafio MBA Engenharia de Software com IA - Full Cycle

Este projeto usa Python, um ambiente virtual e PostgreSQL com a extensão `pgvector`.

## Pré-requisitos

- Python 3.12 ou compatível com o ambiente virtual do projeto ( O Python 3.14 nao funcionou algumas dependencias )
- `pip` e suporte ao módulo `venv`
- Docker e Docker Compose

## 1. Criar e ativar o ambiente virtual

Se o suporte ao `venv` ainda não estiver instalado no seu sistema:

```bash
sudo apt update && sudo apt install python3-full -y
```

Depois, crie o ambiente virtual na raiz do projeto:

```bash
python3 -m venv .venv
```

Ative o ambiente virtual:

```bash
source .venv/bin/activate
```

## 2. Instalar as dependências

Com o ambiente virtual ativo, instale os pacotes do projeto:

```bash
pip install -r requirements.txt
```

## 3. Configurar as variáveis de ambiente

Crie um arquivo `.env` a partir do exemplo do projeto e preencha os valores necessários.

As variáveis usadas pelo código atualmente são:

- `PDF_PATH`
- `PGVECTOR_URL`
- `PGVECTOR_COLLECTION`

## 4. Subir o banco de dados

O projeto já inclui um `docker-compose.yml` com PostgreSQL e `pgvector`.

Suba os containers com:

```bash
docker compose up -d
```

Isso inicia o banco em `localhost:5432` com as credenciais definidas no compose.

## 5. Executar os scripts

Depois de preparar o ambiente, execute os arquivos conforme a etapa desejada:

```bash
python3 src/ingest.py
python3 src/search.py
python3 src/chat.py
```

## Observação

Se algum script retornar erro de inicialização, verifique primeiro as variáveis de ambiente e a conexão com o banco.
O modelo GOOGLE_EMBEDDING_MODEL='models/embedding-001' foi deprecado e removido pelo google. Entao substitui para o modelo mais recente "models/gemini-embedding-001"