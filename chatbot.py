import json
import random
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Load pre-trained model and data
model = load_model('chatbot_trained_model_file.h5')
with open('words.pkl', 'rb') as file:
    words = pickle.load(file)
with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

def preprocess_input(sentence):
    """
    Tokenizes, lemmatizes, and normalizes the input sentence.
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    """
    Converts the input sentence into a bag-of-words representation.
    """
    sentence_words = preprocess_input(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    """
    Predicts the class of the input sentence.
    """
    bow_vector = bow(sentence, words)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    """
    Fetches the appropriate response based on the predicted intent.
    """
    if intents_list:
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I'm sorry, I didn't understand that. Can you please rephrase?"

# Run chatbot
print("GO! Bot is running!")
while True:
    message = input("You: ")
    if message.lower() in ['exit', 'quit']:
        print("Bot: Goodbye!")
        break
    intents_prediction = predict_class(message, model)
    response = get_response(intents_prediction, intents)
    print(f"Bot: {response}")
