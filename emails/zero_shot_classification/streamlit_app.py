import streamlit as st  
import requests  
import pandas as pd  
import numpy as np  
from streamlit_tags import st_tags  
from st_aggrid import AgGrid
from io import BytesIO  


# DEFINE USEFUL FUNCTIONS


def query(payload: dict, api_url: str, headers: dict) -> dict:
    """
    Queries HuggingFace API to access the language model
    and perform zero-shot classification.

    Args:
        payload (dict): dictionary with input and parameters
                        needed to query the HuggingFace API.
        api_url (str): inference URL of the chosen language model.
        headers (dict): headers to be passed when quering an API.

    Returns:
        response (dict): dictionary with API response.
    """
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()


def format_numerals(numerals: list, decimals: int) -> list:
    """
    Accepts a list of scores representing probabilities
    (floating-point numbers in range [0,1]) of text belonging to certain categories.
    Normlizes scores by rounding them to 2 decimal points and multiplying by a 100.

    Args:
        numerals (list): a list of scores.
        decimals (int): number of decimals for rounding the number.

    Returns:
        formatted_values (list): list of probabilities represented as percentages (out of 100%)
    """
    formatted_values = [f"{round(x * 100, decimals)}%" for x in numerals]
    return formatted_values


@st.cache_data
def convert_df(df: pd.DataFrame, file_format: str):
    """
    Takes a DataFrame and stores it in one of the following formats: JSON, CSV, XLS (EXCEL).

    Args:
        df (pd.DataFrame): a DataFrame to store as a file to a local machine.
        file_format (str): chosen file format for downloading.

    Returns:
        file object of the chosen format.

    Raises:
        warning if file format is not supported.

    """
    if file_format == "JSON":
        return df.to_json().encode("utf-8")
    elif file_format == "CSV":
        return df.to_csv(index=False).encode("utf-8")
    elif file_format == "XLS":
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False)
        return excel_buffer.getvalue()
    else:
        st.warning("Unsupported file format selected.")


# MAIN PAGE SETUP


# Set the page configuration
st.set_page_config(
    layout="centered", page_title="Zero-Shot Text Classifier", page_icon="🤗"
)

# Create the logo and heading of the app
logo, heading = st.columns([0.4, 2.35])

with logo:
    st.image("metadata/text-classification.png", width=80)

with heading:
    st.title("Zero-Shot Text Classifier")


# Set up session state to keep the app interactive
if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False


# SIDEBAR SETUP


st.sidebar.write("")

# Field for user to input HuggingFace API key
api_key = st.sidebar.text_input(
    "🤗 Enter your HuggingFace API key:",
    help="Once you created your HuggingFace account, \
    you can get your free API token in your settings page: \
    https://huggingface.co/settings/tokens",
    type="password",
)

# HuggingFace APU inference URL
api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

# Store the API headers as a Python dict
headers = {"Authorization": f"Bearer {api_key}"}

st.sidebar.markdown("---")


# TABS


# Create two tabs:
# 1. main_tab contains user interface for zero-shot classification
# 2. info_tab contains info about the app, technologies used etc.

main_tab, info_tab = st.tabs(["Main", "Info"])

with info_tab:
    st.subheader("Zero-Shot Text Classifier App")

    st.markdown("This app is aimed at providing users with GUI for zero-shot text classification. \
                The app was built using Streamlit and HuggingFace.")
    
    st.subheader("What is Streamlit?")
    
    st.markdown("Streamlit is a Python library which provides tools for creating web applications \
                for ML projects without in-depth comprehension of backend and frontend development. Appart from streamlit, \
                additional libraries which integrate with streamlit apps but must be installed separately were used: \
                streamlit_tags and streamlit_aggrid")

    st.subheader("What is HuggingFace?")
    
    st.markdown("HuggingFace is a platform for Data Scientists providing a wide range of models, \
                datasets, and papers regarding AI. It also provides API which allows programmers to easily \
                integrate models into their projects. ")
    
    st.subheader("Prerequisites")
    
    st.markdown("Before strating with the app one should create a HuggingFace account at https://huggingface.co/ \
                and generate API Access Key at https://huggingface.co/settings/tokens. \
                Enter the key in the sidebar and now you are ready to perform text classification!")
    
    st.subheader("NLP side of the project")
    
    st.markdown("The language model used in the project is the [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli). \
                It uses BART as a base model and was trained on the MultiNLI (MNLI) dataset. \
                Size of the model: 407M params.")

    

with main_tab:
    st.write("")

    # Short project description
    st.markdown(
        """
        
        App for text classification using zero-shot learning.\
        You can use pre-defined classes (email topics) or include your own (up to 10).
        
        """
    )

    st.write()

    # Add custom labels (email categories) and set suggestions (same labels as well)
    with st.form(key="my_form"):
        labels = st_tags(
            value=[
                "Storno",
                "Mahnungen",
                "SIM-Karte Aktivierung",
                "Rechnungen",
                "Tarife",
                "Verträge und Vereinbarungen",
            ],
            maxtags=10,
            suggestions=[
                "Storno",
                "Mahnungen",
                "SIM-Karte Aktivierung",
                "Rechnungen",
                "Tarife",
                "Verträge und Vereinbarungen",
            ],
            label="",
            key="1",
        )

        max_num_texts = 5

        # Default texts for classification. Displayed once the app is initialized.
        # The texts are randomly selected German emails from the given dataset.
        emails_to_classify = [
            "Hallo, mein Name ist Joerg Schulte-Pelkum. Ich war ePlus Kunde bei Ihnen bis zum 16.08.01. \
            Danach erfolgte eine Vertraguebernahme auf Nadie Schulte-Pelkum. Sie haben nun am 19.09.01 \
            fuer die Rechnung am 31.08.01 (Rechnungsnummer 4059839081)den Betrag von 45,26 DM sowohl \
            von meinem Konto als auch von Nadia Schulte-Pelkum abgebucht. Was geht hier vor? MfG,\
            Joerg Schulte-Pelkum",
            "Sehr geehrte Herren, ich habe gestern mit Ihrer Hotline telefoniert.\
            Meine Karte ist defekt und muß ausgetauscht werden. Ich möchte dieses, \
            weil es schneller geht, in einem E-Plus Shop machen. Bitte teilen Sie mir mit, \
            wo es einen Shop in Mühldorf, Altötting oder Vilsbiburg gibt. \
            Ihre Shop-Such-Funktion ist wenig hilfreich. Viele Grüße Christian Ganz",
            """Sehr geehrte Damen und Herren, da ich Siemens-Mitarbeiter bin habe ich über \
            meine Firma einen Rahmenvertrag von Ihnen für Siemensangehörige abgeschlossen. \
            Ich wählte den Time and more 120 Tarif. \
            Dieser Tarif sollte nun eben für Siemensangehörige 37,50 DM anstatt 50 DM kosten. \
            Die ersten paar Monate habe ich diesen Rabatt auch erhalten und auf einen Schlag zahle ich \
            jetzt seit ein paar Monaten 50 DM. Ich bin darüber natürlich mehr als verärgert, \
            wenn ich außerdem noch bedenke wie schlecht zu dem der SMS-Versand bei Ihnen funktioniert. \
            Jede 2 SMS kommt nicht an. Ich möchte Sie bitten mir in Zukunft wieder den "Siemens-Rabatt" \
            einzuräumen und den bis jetzt nicht gewährten Rabatt gutzuschreiben. Vielen Dank im Voraus, Robert Ossiander""",
        ]

        # Convert emails to string and separate them with a newline
        emails_string = "\n".join(map(str, emails_to_classify))

        # Area for the text input
        text = st.text_area(
            "Enter texts to classify:",
            emails_string,
            height=160,
            help=f"You can input up to {str(max_num_texts)} texts for classification. \
                If you need to change the limit, you may fork the repo and tweak the 'max_num_texts' parameter.",
        )

        texts_to_classify = [text for text in text.split("\n")]

        # Put a limit and a warning for the maximum number of texts for classification
        if len(texts_to_classify) > max_num_texts:
            st.info(
                f"Note: you can input up to {str(max_num_texts)} for classification. \
                If you need to change the limit, you may fork the repo and tweak the 'max_num_texts' parameter."
            )

            texts_to_classify = texts_to_classify[:max_num_texts]

        submit_button = st.form_submit_button(label="Classify")

    # CONDITIONAL STATEMENTS
    # filter invalid inputs and check user activity

    match (submit_button, st.session_state.valid_inputs_received, api_key, text, len(labels)):
        case (False, False, _, _, _):
            st.stop()

        # check if the user entered HuggingFace API Key
        case (_, _, "", _, _):
            st.warning("Please, enter your HuggingFace API Key.")
            st.stop()
        
        # check if there are texts to classify
        case (True, _, _, "", _) :
            st.warning("There are no texts to classify.")
            st.session_state.valid_inputs_received = False
            st.stop()

        # check if there are no labels
        case (True, _, _, _, 0) :
            st.warning("Please, add at least two categories for classification.")
            st.session_state.valid_inputs_received = False
            st.stop()

        # check if there is one label only
        case (True, _, _, _, 1) :
            st.warning("Please, add at least one more category for classification.")
            st.session_state.valid_inputs_received = False
            st.stop()

        # executed if everything is correct
        case (True, _, _, _, _) | (_, True, _, _, _):
            if submit_button:
                st.session_state.valid_inputs_received = True

            api_output = []

            # send an API request for classification for each text written by a user
            for text in texts_to_classify:
                json_api_output = query(
                    {
                        "inputs": text,
                        "parameters": {"candidate_labels": labels, "multi_label": True},
                        "options": {"wait_for_model": True},
                    },
                    api_url,
                    headers
                )
                
                api_output.append(json_api_output)

            st.success(":white_check_mark: Done!")

            st.markdown("#### Check the results!")

            # PROCESS DATAFRAME

            df = pd.DataFrame.from_dict(api_output)
            df.rename(columns={"sequence": "Text"}, inplace=True)

            # check the number of labels
            # if the number > 3 display only 3 highest scores and corresponding labels
            if len(labels) > 3:
                df["Label"] = df["labels"].str[:3]
                df["scores"] = df["scores"].apply(lambda x: format_numerals(x, 2))
                df["Score"] = df["scores"].str[:3]

            else:
                df["Label"] = df["labels"]
                df["Score"] = [[f"{x:.2%}" for x in row] for row in df["scores"]]

            df.drop(["labels", "scores"], inplace=True, axis=1)

            df.index = np.arange(1, len(df) + 1)

            # display dataframe in streamlit web app
            AgGrid(df)

            # DOWNLOAD CLASSIFICATION RESULTS

            cs, c1 = st.columns([2, 2])

            with cs:
                # Dropdown for selecting file format
                file_format = st.selectbox("Select file format:", ["JSON", "CSV", "XLS"])
                
                # Download button
                converted_data = convert_df(df, file_format)

                st.download_button(
                    label="Download results",
                    data=converted_data,
                    file_name="classification_results." + file_format.lower(),
                    mime="application/octet-stream",
                )
