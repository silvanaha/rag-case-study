from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

def llm_compare_response_and_reference_documents(response: str, reference: str, model: BaseChatModel) -> str: # evtl reference kein str
    prompt_template = PromptTemplate.from_template(
        """Given reference data market with "reference" and a summary marked with "summary", please judge the relevance of the contents of the summary on a scale of *1* (bad) to *5* (perfect). The more irrelevant information is contained in the summary, the worse the score it will get. Missing or incomplete information must not influence the score.<reference>{reference}</reference> ### <summary>{response}</summary>""")
    prompt = prompt_template.invoke({"response": response, "reference": reference})
    response = model.invoke(prompt)
    print(response)
    return response.content

