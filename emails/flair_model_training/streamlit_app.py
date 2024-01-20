from flair.models import TextClassifier
from flair.data import Sentence
from text_cleaning import preprocess_text
import streamlit as st
import re

# dictionary with category names as values
# and their numerical represantations as keys
categories = {
    "1": "AKTIVIERUNG_SIM",
    "18": "UPGRADE_ERSATZ_GUTSCHRIFT",
    "16": "TARIFE",
    "4": "GERAETE_UND_ZUBEHOER",
    "5": "KUENDIGUNGEN",
    "3": "FREE___EASY",
    "2": "E_PLUS_ONLINE",
    "13": "RECHNUNGEN",
    "11": "NON_VOICE_DIENSTE",
    "8": "MEHRWERTDIENSTE",
    "9": "NETZ",
    "10": "NETZDIENSTE",
    "15": "STORNO",
    "19": "VERTRAEGE_UND_VEREINBARUN",
    "17": "TEILNEHMERSTAMMDATEN",
    "0": "AKTIONEN",
    "6": "KUNDENBETREUUNG_ONLINE",
    "14": "R_KUNDEN",
    "20": "VERTRIEBSPARTNER",
    "12": "PRAEVENTION",
    "7": "MAHNUNGEN",
}


def load_model():
    """
    Load the model from the directory 'flair/model/'
    """
    return TextClassifier.load("flair/model/best-model.pt")


def predict_category(user_email, model):
    """
    Predicts the category of the input email.

    Args:
        user_email (str): input email for category prediction

    Returns:
        formatted_category (str): predicted category
    """
    sentence = preprocess_text(user_email)
    sentence = Sentence(sentence)
    model.predict(sentence)
    predicted_category = sentence.labels[0].value
    category = categories.get(predicted_category, "Uncategorized")
    category_words = re.split("_|___", category)
    formatted_category = " ".join(category_words).capitalize()
    return formatted_category


def main():
    """
    Implements previous functions for model loading and category prediction
    as well as defines the GUI of the streamlit app
    """
    model = load_model()

    st.title("Predict the Category of German Emails using Flair")
    st.markdown(
        "Predicting the category of the input email. \
        The model for this purpose was built on top of the Flair NLP framework."
    )

    user_email = st.text_input("Enter an email: ")
    submit = st.button("Predict the category")

    if submit and user_email:
        predicted_category = predict_category(user_email, model)
        st.write(predicted_category)


if __name__ == "__main__":
    main()
