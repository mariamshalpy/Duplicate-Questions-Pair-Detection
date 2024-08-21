import pickle
import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from keras.models import load_model
import keras.backend as K



def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

def preprocess_text(text, vocabulary, max_seq_length):
    ''' Preprocess and convert text to a list of word indices '''
    text = str(text)
    text = text.lower()
    
    stops = set(stopwords.words('english'))
    # Apply preprocessing steps to the text
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Tokenize text and convert to word indices using vocabulary
    word_indices = [vocabulary[word] if word in vocabulary else 0 for word in text.split() if word not in stops]
    return word_indices

def make_predictions(model, text1, text2, vocabulary, max_seq_length):
    # Preprocess text inputs
    processed_text1 = preprocess_text(text1, vocabulary, max_seq_length)
    processed_text2 = preprocess_text(text2, vocabulary, max_seq_length)

    # Pad sequences
    processed_text1 = pad_sequences([processed_text1], maxlen=max_seq_length)
    processed_text2 = pad_sequences([processed_text2], maxlen=max_seq_length)

    # Make predictions
    predictions = model.predict([processed_text1, processed_text2])[0]

    # Optionally, convert predictions to boolean values
    is_duplicate = (predictions > 0.5)

    return predictions, is_duplicate

def get_model(filepath):
    malstm_model = load_model(filepath)
    return malstm_model

def get_vocab(filepath):
    vocabulary = pickle.load(open(filepath, "rb"))
    return vocabulary



