from unittest import TestCase
from assertpy import assert_that

from src.rag.simple_quality_metrics import compute_rouge


class SimpleQualityMetric(TestCase):

    def test_rouge(self):
        result = compute_rouge("Das ist mein Test, und so weiter, die Mutter hat einen blauen Hund", "Das ist mein Apfelkuchen, deren Mutters Hund blau war", lemmatize=True)
        assert_that(result['rouge2'].recall).is_equal_to(0.375)
        assert_that(result['rouge2'].precision).is_equal_to(0.25)

    def test_rouge_nolemma(self):
        result = compute_rouge("Das ist mein Test, und so weiter, die Mutter hat einen blauen Hund", "Das ist mein Apfelkuchen, deren Mutters Hund blau war", lemmatize=False)
        assert_that(result['rouge2'].recall).is_equal_to(0.25)

    def test_rouge_perfect(self):
        result = compute_rouge("Das ist mein Test, und so weiter", "Das ist mein Testes und so weiter", lemmatize=True)
        assert_that(result['rouge2'].fmeasure).is_equal_to(1)
        assert_that(result['rougeL'].fmeasure).is_equal_to(1)
