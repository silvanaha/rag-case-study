from unittest import TestCase
from assertpy import assert_that

from src.rag.create_rag_pipeline import initialize_rag_pipeline, respond_to_query


class CreateRagPipelineTestCase(TestCase):

    def setUp(self):
        documents_path = "testdata"
        retriever = "faiss"
        self.model, self.retriever, self.documents = initialize_rag_pipeline(documents_path, retriever)
        assert_that(len(self.documents)).is_equal_to(10)
        assert_that(self.model).is_not_none()
        assert_that(self.retriever).is_not_none()

    def test_do_rag(self):
        user_question = "Ist Deixis ein Konzept in der Pragmatik?"

        response = respond_to_query(user_question, self.retriever, self.model)
        print(f"response to {user_question}:/n {response}")
        assert_that(response).is_not_none()


    def test_do_rag_oov(self):
        user_question = "Welche Rezepte kennst du für Kartoffeln?"

        response = respond_to_query(user_question, self.retriever, self.model)
        print(f"response to {user_question}:/n {response}")
        assert_that(response).is_not_none()