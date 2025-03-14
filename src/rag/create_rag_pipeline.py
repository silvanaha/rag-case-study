from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_core.language_models import BaseChatModel

from src.rag.create_index_and_retrieve import retrieve_results, get_ensemble_retriever_from_documents, \
    get_faiss_vectorstore_retriever_from_documents, get_bm25_retriever_from_documents
from src.rag.custom_xml_loader import CustomXMLLoader
from src.rag.read_xml_files import list_file_paths
from src.rag.similarity_reranker import filter_results_by_similarity_cutoff


def init_model() -> BaseChatModel:
#    return init_chat_model("mistral-large-latest", model_provider="mistralai")
    return init_chat_model("open-mistral-7b", model_provider="mistralai")

def create_prompt_template():
    template = "Based on the following documents: {context}, answer the question: {question}"
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt_template

def create_context(results):
    context = " ".join([x.page_content for x in results]) ## TODO add source
    return context

def respond_to_query(user_question:str, retriever: BaseRetriever, model: BaseChatModel):

    results = retrieve_results(user_question, retriever)
    for document in results:
        print(document.page_content)
    print(f"before filtering {len(results)}")
    filtered = filter_results_by_similarity_cutoff(user_question, results, 0.8)
    print(f"after filtering {len(filtered)}")
    for document in filtered:
        print(document.page_content)
    context = create_context(filtered)
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

def initialize_rag_pipeline(documents_path: str, retriever_selection: str = "ensemble"):
    model = init_model()
    documents = load_all_xml_documents(documents_path)
    if retriever_selection == "bm25":
        retriever = get_bm25_retriever_from_documents(documents, source_id=1, num_results=7)
    elif retriever_selection == "faiss":
       retriever = get_faiss_vectorstore_retriever_from_documents(documents, source_id=1, num_results=7)
       return model, retriever, documents
    else:
        retriever = get_ensemble_retriever_from_documents(documents, source_id=1, num_results=7)

    return model, retriever, documents


