from rouge_score import rouge_scorer
import spacy



def compute_rouge(response: str, reference_documents: str, lemmatize=False) -> {}:
    scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'])
    if lemmatize:
        nlp = spacy.load("de_core_news_sm")
        reference_nlp = nlp(reference_documents)
        response_nlp = nlp(response)
        lemmatized_response = [word.lemma_ for word in response_nlp]
        lemmatized_reference = [word.lemma_ for word in reference_nlp]
        return scorer.score(target=" ".join(lemmatized_reference), prediction=" ".join(lemmatized_response))
    else:
        return scorer.score(target=reference_documents, prediction=response)

