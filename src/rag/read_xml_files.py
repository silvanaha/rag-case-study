import os
import glob
from pathlib import Path

import untangle
from untangle import Element


def list_file_paths(path_to_files: str) -> [str]:
    file_paths: [str] = []
    if Path(path_to_files).is_dir():
#        return list((Path(path_to_files)).glob(f"{path_to_files}/*.xml"))
        for file_path in glob.iglob(f"{path_to_files}/*.xml"):
            file_paths.append(file_path)
        return file_paths
    else:
        raise FileNotFoundError


def parse_detangled_input_files(parsed: Element, add_metadata: bool = False):
    field = parsed.document.metadata.field
    documents: [str] = []
    keywords = [keyword.cdata for keyword in parsed.document.metadata.keywords]
    for chapter in parsed.document.content.chapter:
        chapter_name = chapter["class"]
        try:
            for section in chapter.section:
                section_title = section["title"]
                print(section_title)
                for paragraph in section.paragraph:
                    document = extract_content_from_paragraph(paragraph, add_metadata, chapter_name, field, keywords,
                                                              section_title)
                    documents.append(document)
        except AttributeError:
            for paragraph in chapter.paragraph:
                document = extract_content_from_paragraph(paragraph, add_metadata, chapter_name, field, keywords)
                documents.append(document)
    return documents


def extract_content_from_paragraph(paragraph, add_metadata, chapter_name, field, keywords, section_title=None) -> [str]:
    document = paragraph.cdata
    document += " " + chapter_name
    if section_title:
        document += ", " + section_title
    if add_metadata:
        document += ", " + field.cdata
        document += ", " + ", ".join(keywords)
    return document


def parse_xml_files(path_to_file: str, add_metadata: bool = False) -> [str]:
    untangled_xml = untangle.parse(path_to_file)
    documents = parse_detangled_input_files(untangled_xml, add_metadata)
    return documents