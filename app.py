import streamlit as st
import pyttsx3
import tensorflow as tf
import torch
import random as rd
import numpy as np
from transformers import BertModel, BertTokenizer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
model = tf.keras.models.load_model('Final_Model_GRU')
class_labels =['admission', 'canteen', 'course', 'creator', 'event', 'facilities',
       'fees', 'goodbye', 'greeting', 'hod', 'hostel', 'hours', 'library',
       'location', 'name', 'number', 'placement', 'principal', 'ragging',
       'random', 'salutation', 'scholarship', 'seats', 'sem', 'size',
       'sports', 'task', 'uniform', 'vacation']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model1 = BertModel.from_pretrained('bert-base-uncased')

def extract_bert_features(tokenized_input):
    features = []
    encoded_input = tokenizer.encode_plus(
        tokenized_input,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=14,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model1(**encoded_input)
    sentence_features = outputs.last_hidden_state.squeeze(0).numpy()
    features.append(sentence_features)
    return features
import re
from nltk.corpus import stopwords

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\d+", "NUM", text)
    
    stop_words = set(stopwords.words("english"))

    custom_stopwords = ["how", "are", "you", "other", "question", "phrases"]
    stop_words.difference_update(custom_stopwords)

    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }
    tokens = text.split()
    processed_tokens = [contractions.get(word, word) for word in tokens]
    processed_text = " ".join(processed_tokens)

    return processed_text

def fun1(text):
    tokens = word_tokenize(text)
    lm = WordNetLemmatizer()
    lemmatized_tokens = [lm.lemmatize(token) for token in tokens]
    return lemmatized_tokens

with open('intents2.json', 'r') as file:
    data = json.load(file)

def main():
    st.set_page_config(page_title="AI Chatbot", page_icon="aibot.png") 

    st.image("aibot.png", use_column_width=True)
    st.title("AI CHATBOT")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    user_input = st.text_input("You: ")

    if user_input.lower() == "quit":
        st.stop()

    if user_input:
        bot_response = chat_with_bot(user_input, st.session_state.conversation_history)
        st.session_state.conversation_history.append(("You", user_input))
        st.session_state.conversation_history.append(("Bot", bot_response))

        engine = pyttsx3.init()
        engine.setProperty('rate', 150)

        engine.say(bot_response)
        engine.runAndWait()

    st.text_area("Chat History", value="\n".join([f"{sender}: {message}" for sender, message in st.session_state.conversation_history]))

def chat_with_bot(user_input, conversation_history):
    preprocessed_input = preprocess_text(user_input)
    processed_input = fun1(preprocessed_input)
    features = extract_bert_features(processed_input)
    features = features[0]
    features = np.reshape(features, (1, 14, 768))
    test = model.predict(features)
    predicted_label = np.argmax(test)
    predicted_class_name = class_labels[predicted_label]
    prediction_accuracy = test[0][predicted_label]

    if prediction_accuracy >= 0.8:
        for x in data['intents']:
            if x['tag'] == predicted_class_name:
                responses = x['responses']
                if isinstance(responses, list):
                    bot_response = rd.choice(responses)
                else:
                    bot_response = responses

                if isinstance(bot_response, list):
                    bot_response = ' '.join(bot_response)
                return bot_response
    else:
        bot_response = "Sorry, I didn't understand you.Can u rephrase the question"
        return bot_response

if __name__ == "__main__":
    main()