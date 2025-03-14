
from unittest import TestCase
from assertpy import assert_that
from langchain_mistralai import ChatMistralAI
from src.rag.llm_quality_metrics import llm_compare_response_and_reference_documents

class LLMQualityMetric(TestCase):

    def setUp(self):
        self.model = ChatMistralAI()
        self.reference = """Die Pragmatik ist ein Teilgebiet der Linguistik.
        Die Pragmatik beschäftigt sich mit kontextabhängigen Aspekten der sprachlichen Bedeutung.
        Der Begriff  „Pragmatik" leitet sich ab vom altgriechischen πρᾶγμα"""

    def test_llm_compare(self):
        fake_response_bad = "Die pragmatische Lösung wäre schneller: Kuchen vom Bäcker"
        comparison_output = llm_compare_response_and_reference_documents(fake_response_bad, self.reference, self.model)
        assert_that(comparison_output).is_not_none()

    def test_llm_compare_good(self):
        fake_response_good = "Die Pragmatik ist ein Teilgebiet der Linguistik und beschäftigt sich mit kontextuellen Aspekten, wie beispielsweise Deixis."
        comparison_output = llm_compare_response_and_reference_documents(fake_response_good, self.reference, self.model)
        assert_that(comparison_output).is_not_none()
