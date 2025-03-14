import streamlit as st

from src.rag.create_rag_pipeline import respond_to_query, initialize_rag_pipeline
from src.rag.llm_quality_metrics import llm_compare_response_and_reference_documents
from src.rag.simple_quality_metrics import compute_rouge

st.set_page_config(
    page_title="DEV DEMO",
    page_icon=":eye:",
)

st.title('MEDICAL RAG DEV DEMO :computer:')
st.markdown(f"""Die DEV DEMO gibt Detailinformationen anhand der gegebenen Nutzer-Anfrage: 
* Sie zeigt Zwischenergebnisse aus dem Retrieval, 
* Evaluationsmetriken am Beispiel von Rouge,
* eine erster POC für eine LLM-basierte Evaluation, die die Systemausgabe mit einer Referenz vergleicht. """)
st.markdown("⚙️ Hinweis zur Nutzung: Die Basis dieses KI-Systems ist unsere umfassende, aktuelle medizinische Experten-Datenbank. "
            "Das System ist momentan noch in Entwicklung. Wir empfehlen die Überprüfung der Aussagen und freuen uns  über Nutzer-Feedback.")


st.sidebar.header("Datenbank Setup :brain:")
data_path = st.sidebar.text_input(label="Pfad zu Referenzdaten:", value="./data/documents/")


@st.cache_resource
def create_index_st(path_to_data):
    texts = ["Das ist mein Sample Text", "Das ist mein zweiter inhaltsvoller Satz mit Kartoffeln",
         "Rote Bete ist erstaunlich lecker"]
    (modell, retrievely, documenty) = initialize_rag_pipeline(documents_path=path_to_data, retriever_selection="faiss")
    return modell, retrievely, texts


st.sidebar.markdown("_Lade Datenbank von..._")
st.sidebar.write(data_path)
model, retriever, documents = create_index_st(data_path)
st.sidebar.write("_...fertig_")

st.header("User Interaction")
with st.form("query_form"):
    user_question = st.text_area(
        "Bitte geben Sie hier Ihre Frage ein, zum Beispiel:",
        "Was sind die verschiedenen Symptome von Aphasie?",
    )
    submitted = st.form_submit_button("Los gehts")
if submitted:
    st.markdown("_Vielen Dank für die Frage! Die Antworten sind unterwegs_ 🧬 ")
    response, (prompt, filtered, filtered_scores) = respond_to_query(user_question, retriever, model)
    st.session_state['tricks'] = response.content
    st.write(response.content)

    st.subheader("Retrieved Documents")
    st.markdown("The documents are shown with embeddings similarity scores that were used to filter the results based on a cutoff score:")
    for i in range(len(filtered)):
        st.markdown(f"{filtered_scores[i]}\t_{filtered[i].metadata['source']}_: {filtered[i].page_content}")

    st.subheader("Prompt")
    st.markdown(prompt)

    st.subheader("Rouge Score")
    st.markdown("Computed between response and reference documents:")
    rouge_score = compute_rouge(response.content, "\n".join([x.page_content for x in filtered]) , lemmatize=True)
    st.markdown(f"rouge2 {rouge_score['rouge2'].precision:.4f} {rouge_score['rouge2'].recall:.4f} {rouge_score['rouge2'].fmeasure:.4f}")
    st.markdown(f"rougeL {rouge_score['rougeL'].precision:.4f} {rouge_score['rougeL'].recall:.4f} {rouge_score['rougeL'].fmeasure:.4f}")

    st.subheader("LLM Evaluation Score")

with st.form("eval_form"):
    reference2 = st.text_area(
        "Please write a reference answer here", 
    )
    submitted2 = st.form_submit_button("Evaluate")
if submitted2:

    comparison_output = llm_compare_response_and_reference_documents(st.session_state.tricks, reference2, model)
    st.markdown(comparison_output)
"""
Dev Output
* show prompt 
* show documents after retrieval
* compute rouge scores (using documents as reference)
* compute LLM quality score (using input as reference)

"""