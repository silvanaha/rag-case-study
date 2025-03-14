from unittest import TestCase
from assertpy import assert_that

from src.rag.create_index_and_retrieve import get_bm25_retriever_from_text, get_ensemble_retriever_from_texts, \
    get_faiss_vectorstore_retriever_from_text, retrieve_results, get_ensemble_retriever_from_documents


class CreateIndexTestCase(TestCase):


    def test_create_bm25retriever(self):
        texts = ["Das ist mein Sample Text", "Das ist mein zweiter inhaltsvoller Satz", "Rote Bete ist erstaunlich lecker"]
        retriever = get_bm25_retriever_from_text(texts, source_id=111, num_results=2)
        assert_that(retriever.get_name()).is_equal_to("BM25Retriever")
        assert_that(len(retriever.docs)).is_equal_to(3)
        assert_that(retriever.docs[0].metadata["source"]).is_equal_to(111)

    def test_create_fass_vectorstore_retriever(self):
        texts = ["Das ist mein Sample Text", "Das ist mein zweiter inhaltsvoller Satz", "Rote Bete ist erstaunlich lecker"]
        retriever = get_faiss_vectorstore_retriever_from_text(texts, source_id=112, num_results=2)
        assert_that(retriever.get_name()).is_equal_to("VectorStoreRetriever")
        assert_that(retriever.vectorstore.index.ntotal).is_equal_to(3)

    def test_get_ensemble_retriever(self):
        texts = ["Das ist mein Sample Text", "Das ist mein zweiter inhaltsvoller Satz", "Rote Bete ist erstaunlich lecker"]
        retriever = get_ensemble_retriever_from_texts(texts, source_id=112, num_results=2)
        assert_that(retriever).is_not_none()
        assert_that(len(retriever.retrievers)).is_equal_to(2)
        assert_that(retriever.weights[0]).is_equal_to(0.5)

    def test_query_retriever_with_ensemble(self):
        texts = ["Das ist mein Sample Text", "Das ist mein zweiter inhaltsvoller Satz", "Rote Bete ist erstaunlich lecker"]
        query= "Gib mir Infos zu leckerem Inhalt"
        retriever = get_ensemble_retriever_from_texts(texts, source_id=1, num_results=2)
        results = retrieve_results(query, retriever)
        assert_that(results).is_not_none()
        assert_that(len(results)).is_equal_to(3)
        assert_that(results[0].page_content).is_equal_to(texts[1])
        assert_that(results[1].page_content).is_equal_to(texts[2])

    def test_query_retriever_with_faiss(self):
        texts = ["Das ist mein Sample Text", "Das ist mein zweiter inhaltsvoller Satz", "Rote Bete ist erstaunlich lecker"]
        query= "Gib mir Infos zu leckerem Inhalt"
        retriever = get_faiss_vectorstore_retriever_from_text(texts, source_id=1, num_results=2)
        results = retrieve_results(query, retriever)
        assert_that(results).is_not_none()
        assert_that(results[0].page_content).is_equal_to(texts[1])
        assert_that(results[1].page_content).is_equal_to(texts[0])

    def test_query_retriever_with_bm25(self):
        texts = ["Das ist mein Sample Text", "Das ist mein zweiter inhaltsvoller Satz", "Rote Bete ist erstaunlich lecker"]
        query= "Gib mir Infos zu leckerem Inhalt"
        retriever = get_bm25_retriever_from_text(texts, source_id=1, num_results=2)
        results = retrieve_results(query, retriever)
        assert_that(results).is_not_none()
        assert_that(results[0].page_content).is_equal_to(texts[2])
        assert_that(results[1].page_content).is_equal_to(texts[1])




