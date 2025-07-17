# PlottingCode
Python Script for Plotting data for pyschology experiments

# RAG Pipeline for Log Summarization with Gemini 2.0 + LangChain

import os
import json
import time
from pathlib import Path
from typing_extensions import TypedDict, List
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langgraph.graph import START, StateGraph

# ────────────── API Key ───────────────
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

# ────────────── LLM (Gemini 2.0) ───────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# ────────────── Embeddings & Vector Store ───────────────
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# ────────────── Log Loader ───────────────
def load_logs_from_folder(folder_path):
    docs = []
    for file_path in Path(folder_path).glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                cleaned = line.strip()
                if cleaned:
                    docs.append(Document(
                        page_content=cleaned,
                        metadata={"source": file_path.name, "line": idx + 1}
                    ))
    return docs

# ────────────── Ingest and Index ───────────────
log_docs = load_logs_from_folder("./logs")
vector_store.add_documents(documents=log_docs)

# ────────────── Prompt Template ───────────────
prompt_summarize_logs = PromptTemplate(
    input_variables=["context", "question"],
    template=r"""
You are a log analysis assistant.

Task → Summarize the following log entries and extract patterns or call flows.

Include:
1. Main issues or events
2. Recurring patterns or errors
3. Any identifiable call flow

Respond clearly and concisely using ONLY the context.

Context:
{context}

Question:
{question}

Return your summary in plain text.
"""
)

# ────────────── Retriever ───────────────
bm25_retriever = BM25Retriever.from_documents(log_docs)
bm25_retriever.k = 8
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_store.as_retriever(search_kwargs={"k": 8})],
    weights=[0.5, 0.5],
)

# ────────────── LangGraph State ───────────────
class Search(BaseModel):
    query: str = Field(description="Search query to run.")

class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

# ────────────── Functions ───────────────
def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    return {"query": structured_llm.invoke(state["question"])}

def retrieve(state: State):
    query = state["query"]
    return {"context": ensemble_retriever.invoke(query.query)[:10]}

def generate_summary(state: State):
    docs_content = "\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_summarize_logs.format(context=docs_content, question=state["question"])
    response = llm.invoke(prompt)
    return {"answer": response.content}

# ────────────── Build Graph ───────────────
summarize_graph = StateGraph(State).add_sequence([
    analyze_query,
    retrieve,
    generate_summary
])
summarize_graph.add_edge(START, "analyze_query")
graph_summarize = summarize_graph.compile()

# ────────────── Test with Dummy Logs ───────────────
# Create dummy logs (only if no real logs exist)
dummy_log_path = Path("./logs/dummy_log.txt")
dummy_log_path.parent.mkdir(exist_ok=True)
if not dummy_log_path.exists():
    dummy_log_path.write_text("""
[INFO] 10:00:01 Service A started
[DEBUG] 10:00:02 Initializing modules...
[INFO] 10:00:03 Module X initialized
[ERROR] 10:00:04 Failed to connect to Service B
[INFO] 10:00:05 Retrying connection...
[INFO] 10:00:06 Connection successful
[INFO] 10:00:07 Service A completed request
""", encoding="utf-8")

# Run test query
question = "Summarize log behavior and detect patterns."
result = graph_summarize.invoke({"question": question})
print("Summary:\n", result["answer"])
print("\nContext:\n", result["context"])
