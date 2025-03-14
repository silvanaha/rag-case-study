import streamlit as st

from src.rag.create_rag_pipeline import init_model, respond_to_query, initialize_rag_pipeline

st.title('MEDICAL RAG DEMO :health_worker:')
st.markdown("Das Medical Rag System informiert Fachleute und Laien über Themen aus der Fach- und Allgemeinmedizin. "
            "Dabei dient es als Sparringspartner bei der medizinischen Diagnostik und unterstützt im Dialog bei der Aus- und Weiterbildung. ")
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
model, retriever, texts = create_index_st(data_path)
st.sidebar.write("_...fertig_")

with st.form("query_form"):
    user_question = st.text_area(
        "Bitte geben Sie hier Ihre Frage ein, zum Beispiel:",
        "Was sind die verschiedenen Symptome von Aphasie?",
    )
    submitted = st.form_submit_button("Los gehts")
    if submitted:
        st.markdown("_Vielen Dank für die Frage! Die Antworten sind unterwegs_ 🧬 ")
        response = respond_to_query(user_question, retriever, model)
        st.write(response.content)
