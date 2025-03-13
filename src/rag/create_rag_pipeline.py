from assertpy import assert_that
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_mistralai import MistralAIEmbeddings
from transformers import AutoModel, AutoTokenizer
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS

from langchain_core.language_models import BaseChatModel

from src.rag.create_index_and_retrieve import retrieve_results, get_ensemble_retriever_from_documents, \
    get_faiss_vectorstore_retriever_from_documents
from src.rag.custom_xml_loader import CustomXMLLoader
from src.rag.read_xml_files import list_file_paths


def init_model() -> BaseChatModel:
    return init_chat_model("mistral-large-latest", model_provider="mistralai")

def create_prompt_template():
    template = "Based on the following documents: {context}, answer the question: {question}"
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt_template

def create_context(results):
    context = " ".join([x.page_content for x in results])
    return context

# TODO do I need to pass on the full texts or can I get them out of the index?
def respond_to_query(user_question:str, retriever: BaseRetriever, model: BaseChatModel):

    results = retrieve_results(user_question, retriever)
    context = create_context(results)
    prompt_template = create_prompt_template()
    prompt = prompt_template.invoke({"context": {context}, "question": {user_question}})
    response = model.invoke(prompt)
    return response

def load_all_xml_documents( path_to_files):

    documents: [Document] = []
    for path in list_file_paths(path_to_files):
        loader = CustomXMLLoader(path)
        documents += loader.load()
    return documents

def initialize_rag_pipeline(documents_path: str):
    model = init_model()
    documents = load_all_xml_documents(documents_path)

#    retriever = get_ensemble_retriever_from_documents(documents, source_id=1, num_results=2) # TODO type
    retriever = get_faiss_vectorstore_retriever_from_documents(documents, source_id=1, num_results=2) # TODO type
    return model, retriever, documents

def do_rag(query, model, retriever):
    response = respond_to_query(query, retriever, model)
    return response