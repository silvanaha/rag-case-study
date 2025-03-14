from unittest import TestCase
from assertpy import assert_that
from langchain_core.documents import Document

from src.rag.similarity_reranker import get_similarity_scores_for_query, rerank_results_for_query, \
    filter_results_by_similarity_cutoff


class RerankResults(TestCase):

    def test_get_similarity_scores(self):
        query = "Ist Deixis ein Konzept in der Pragmatik?"
        results = """Ist Deixis ein Konzept in der Pragmatik?
                Die Pragmatik beschäftigt sich mit kontextabhängigen Aspekten der sprachlichen Bedeutung. Einführung
                Der Begriff  „Pragmatik" leitet sich ab vom altgriechischen πρᾶγμα. Einführung
                Die Pragmatik ist ein Teilgebiet der Linguistik. Einführung""".split("\n")
        scored = get_similarity_scores_for_query(query, results, embedding_model="sentence-transformer")
        assert_that(scored).is_not_none()
        assert_that(len(scored)).is_equal_to(len(results))
        assert_that(scored[0]).is_equal_to(1)

    def test_rerank_results(self):
        query = "Ist Deixis ein Konzept in der Pragmatik?"
        results = """Die Pragmatik beschäftigt sich mit kontextabhängigen Aspekten der sprachlichen Bedeutung. Einführung
Der Begriff  „Pragmatik" leitet sich ab vom altgriechischen πρᾶγμα. Einführung
Ist Deixis ein Konzept in der Pragmatik?
Die Pragmatik ist ein Teilgebiet der Linguistik. Einführung
Konversationelle Implikaturen sind ein wichtiges Thema. Methoden der Pragmatik, Relevante Begriffe""".split("\n")
        expected = """Ist Deixis ein Konzept in der Pragmatik?
Die Pragmatik beschäftigt sich mit kontextabhängigen Aspekten der sprachlichen Bedeutung. Einführung
Der Begriff  „Pragmatik" leitet sich ab vom altgriechischen πρᾶγμα. Einführung
Konversationelle Implikaturen sind ein wichtiges Thema. Methoden der Pragmatik, Relevante Begriffe
Die Pragmatik ist ein Teilgebiet der Linguistik. Einführung""".split("\n")

        reordered = rerank_results_for_query(query, results, embedding_model="sentence-transformer")
        assert_that(reordered[0]).is_equal_to(expected[0])
        assert_that(reordered[2]).is_equal_to(expected[2])
        assert_that(reordered[-1]).is_equal_to(expected[-1])

    def test_filter_results_mistral(self):
        query = "Ist Deixis ein Konzept in der Pragmatik?"
        results = [
            Document("Die Pragmatik beschäftigt sich mit kontextabhängigen Aspekten der sprachlichen Bedeutung. Einführung"),
            Document("Ist Deixis ein Konzept in der Pragmatik?"),
            Document("Die Pragmatik ist ein Teilgebiet der Linguistik. Einführung"),
            Document("Konversationelle Implikaturen sind ein wichtiges Thema. Methoden der Pragmatik, Relevante Begriffe")
        ]
        filtered = filter_results_by_similarity_cutoff(query, results, embedding_model="mistral", cutoff=0.84)
        assert_that(len(filtered)).is_equal_to(3)

    def test_filter_results_sentence_transformer(self):
        query = "Ist Deixis ein Konzept in der Pragmatik?"
        results = [
            Document("Die Pragmatik beschäftigt sich mit kontextabhängigen Aspekten der sprachlichen Bedeutung. Einführung"),
            Document("Ist Deixis ein Konzept in der Pragmatik?"),
            Document("Die Pragmatik ist ein Teilgebiet der Linguistik. Einführung"),
            Document("Konversationelle Implikaturen sind ein wichtiges Thema. Methoden der Pragmatik, Relevante Begriffe")
        ]
        filtered = filter_results_by_similarity_cutoff(query, results, embedding_model="sentence-transformer", cutoff=.84)
        assert_that(len(filtered)).is_equal_to(2)

