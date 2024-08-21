import streamlit as st
from predictor import make_predictions, get_model, get_vocab
import keras.backend as K

import nltk

# Ensure the stopwords resource is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Rest of your code


def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

vocabulary = get_vocab('vocabulary_train.p')
malstm_model = get_model('malstm_trained.h5')
max_seq_length = 212

def main():
    st.title("Duplicate Questions Pair Detection")

    question1 = st.text_input("Enter Question 1:")
    question2 = st.text_input("Enter Question 2:")

    if st.button("Check for Duplicates"):
        

        # Predict duplicate
        prediction, is_duplicate = make_predictions(malstm_model, question1, question2, vocabulary, max_seq_length)

        if is_duplicate:
            st.write("The questions are duplicates!")
        else:
            st.write("The questions are not duplicates.")


if __name__ == "__main__":
    main()
