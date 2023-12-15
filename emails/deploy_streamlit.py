import streamlit as st
from training_script import models, X_train_tfidf, y_train, count_vect

clf = models[1].fit(X_train_tfidf, y_train)

st.title("German Email Category Detection")
st.markdown("Predicting the category of the input email (Rechnungen, Mahnungen, Tarife, Sim Aktivierung, Storno, Vertraege) using LinearCSV.")

message = st.text_input('Enter an email:')

submit = st.button('Predict')

if submit:

    prediction = clf.predict(count_vect.transform([message]))

    print(prediction)

    st.write(f"Predicted category: {prediction[0].capitalize()}.")

# streamlit run app.py 