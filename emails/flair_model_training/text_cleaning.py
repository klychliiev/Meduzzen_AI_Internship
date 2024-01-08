import re
import spacy

# load German-language model for stop words removal
de_nlp = spacy.load("de_core_news_sm")


def fetch_body(text: str) -> str:
    """
    Fetches email body (the message itself) from each email.
    All the emails have similar structure, starting after the word 'Nachricht'
    and ending before the multiple dash characters.

    Args:
        text (str): input unprocessed German email

    Returns:
        text (str): email body with no newlines and extra spaces
    """
    text = text.split("Nachricht      : ")[-1]
    text = text.split("--")[0]
    text = re.sub("\n", " ", text)
    text = re.sub(" +", " ", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return text.strip()


def preprocess_text(text: str) -> str:
    """
    Function for text preprocessing. 
    Includes the following steps: 
    1. lowercase conversion
    2. special characters, punctuation and stopwords removal, 
    3. lemmatization, 
    4. deletion of artefacts left after preprocessing (dashes and extra spaces)
    """
    text = text.lower()
    text = re.sub(r"^a-zäöüß0-9\s", "", text)
    text = re.sub(r"\s", " ", text).strip()
    doc = de_nlp(text)
    no_stopwords_sent = [str(word.lemma_).lower() for word in doc if not word.is_stop]
    text = " ".join(no_stopwords_sent)
    text = re.sub("--", "", text.replace("\n", ""))
    text = re.sub(" +", " ", text)
    return text
