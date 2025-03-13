import unittest
from unittest import TestCase
from assertpy import assert_that

from src.rag.read_xml_files import list_file_paths, parse_xml_files


class ReadXMLTestCase(TestCase):


    def test_read_xml_files(self):
        path = "testdata"
        file_paths = list_file_paths(path)
        assert_that(len(file_paths)).is_equal_to(2)
        assert_that(file_paths).contains("testdata/sample_2.xml")


    def test_parse_xml_file(self):
        file_path = "testdata/sample_2.xml"
        documents = parse_xml_files(file_path, add_metadata=True)
        assert_that(len(documents)).is_equal_to(7)
        expected = "Die Pragmatik ist ein Teilgebiet der Linguistik. Einführung, Pragmatik, Linguistik, Semantic"
        assert_that(documents[0]).is_equal_to(expected)

    def test_parse_xml_file(self):
        file_path = "testdata/sample_2.xml"
        documents = parse_xml_files(file_path, add_metadata=False)
        assert_that(len(documents)).is_equal_to(7)
        expected = "Die Pragmatik ist ein Teilgebiet der Linguistik."
        assert_that(documents[0]).is_equal_to(expected)

    @unittest.skip("local data")
    def test_parse_xml_file(self):
        file_path = "/home/silvana/projects/rag_case_study/data/documents/Aphasie.xml"
        documents = parse_xml_files(file_path, add_metadata=True)
        assert_that(len(documents)).is_equal_to(33)
        expected = "Eine Aphasie (altgriechisch ἀφασία aphasía ‚Sprachlosigkeit') ist eine erworbene Störung der Sprache aufgrund einer Beschädigung (Läsion) von bestimmten Regionen des Gehirns, die für die Steuerung der Sprache entscheidend sind. Einführung, Neurologie, Aphasie, Sprachstörung, Broca-Aphasie, Wernicke-Aphasie, Globale Aphasie, Amnestische Aphasie, Sprachtherapie, Hirnläsion"
        assert_that(documents[0]).is_equal_to(expected)