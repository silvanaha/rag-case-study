import torch
import numpy as np
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from scipy import spatial
from transformers import AutoTokenizer, AutoModel



def create_embeddings(texts: [str], embedding_type ="sentence-transformer") -> np.ndarray :
    if embedding_type == "sentence-transformer":
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        inputs = tokenizer(text=texts, return_tensors="pt", padding=True, truncation = True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state[:,0,:]
        embeddings = embeddings.numpy()
        return embeddings
    else:
        embedding = MistralAIEmbeddings(model="mistral-embed")
        embeddings = np.array(embedding.embed_documents(texts))
        return embeddings


def rerank_results_for_query(query: str, results: [str], embedding_model: str):
    scores = get_similarity_scores_for_query(query, results, embedding_model)
    scored_dict = dict(zip(scores, results))
    reordered = []
    for k in sorted(scored_dict, reverse=True):
        print(k, scored_dict[k])
        reordered.append(scored_dict[k])
    return reordered

def filter_results_by_similarity_cutoff(query: str, results : [Document], embedding_model: str = "mistral-ai", cutoff: float = 0.8) -> [Document]:
    texts = [x.page_content for x in results]
    scores = get_similarity_scores_for_query(query, texts, embedding_model)
    reranked = rerank_results_for_query(query, texts, embedding_model)
    scored_dict = dict(zip(scores, results))
    filtered = [scored_dict[score] for score in scored_dict if score > cutoff]
    return filtered


def get_similarity_scores_for_query(query: str, results: [str], embedding_model: str) -> [str]:
    all_texts = [query]
    all_texts += results
    embeddings = create_embeddings(texts=all_texts, embedding_type=embedding_model)
    query_embedding = embeddings[0]
    scored = []
    for embedding in embeddings[1:]:
        score = 1 - spatial.distance.cosine(embedding, query_embedding)
        scored.append(score)
    return scored
