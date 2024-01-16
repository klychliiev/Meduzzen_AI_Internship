import streamlit as st
from linear_svc_classifier import clf, count_vect


def main():
    """
    Create a web app using streamlit to predict email category using linear SVC.
    """

    st.title("German Email Category Detection")
    st.markdown("Predicting the category of the input email (Rechnungen, Mahnungen, Tarife, Sim Aktivierung, Storno, Vertraege) using LinearCSV.")

    message = st.text_input('Enter an email:')

    submit = st.button('Predict')

    if submit:
        prediction = clf.predict(count_vect.transform([message]))
        st.write(f"Predicted category: {prediction[0].capitalize()}.")


if __name__=='__main__':
    main()
    