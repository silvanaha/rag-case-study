from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_mistralai import MistralAIEmbeddings
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS



model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_bm25_retriever(texts: [str], source_id: int, num_results: int) -> BM25Retriever:
    # TODO additional metadata?
    retriever = BM25Retriever.from_texts(texts, metadatas=[{"source": source_id}] * len(texts))
    retriever.k = num_results
    return retriever

def get_faiss_vectorstore_retriever(texts: [str], source_id: int, num_results: int) -> VectorStoreRetriever:
    embedding = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = FAISS.from_texts(texts, embedding, metadatas=[{"source": source_id}] * len(texts))
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": num_results})
    return faiss_retriever

def get_ensemble_retriever(texts: [str], source_id: int, num_results: int) -> EnsembleRetriever:
    # TODO restrict results NO?
    return EnsembleRetriever(retrievers=[get_bm25_retriever(texts, source_id, num_results),
                                         get_faiss_vectorstore_retriever(texts, source_id, num_results)],
                             weights=[0.5,0.5])

def run_query_on_index(textual_query, retriever) -> ([float], [int]):
    results = retriever.invoke(textual_query)
    return results


