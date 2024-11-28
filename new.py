# this file is used for tokenization of the data. Tokens are the building blocks of the used for analysis, training or processing by AI models.
# The tokenization is done using the nltk library. The nltk library is a powerful library that is used for natural language processing.
# it helps in breaking down data into words, subwords or characters example: "Ai is amazing" -> ["Ai", "is", "amazing"].
# The tokenization is done using the word_tokenize function from the nltk library.
# for code files, it separates keywords, variables and symbols. example: int x= 10; -> ["int", "x", "=", "10", ";"]

#the file is also used to LEMMATIZE the data. Lemmatization is the process of converting a word to its base form. example: "running" -> "run"
#this helps make text easier to analyze by grouping different forms of a word into one. 
#the lemmatization is done using the WordNetLemmatizer from the nltk library.
# we Lematize data so that to ensure 1) standardization it ensures that words like 'running', 'rans' abd 'runs' are treated as one. 
# 2) make it easier to find patterns in the data.
# 3) reduce the size of the data so that models can focus on meanings rather than variations. 
# how it workds ? words get analyzed for their grammatical role (like nouns and verbs) to find the lemma. example "running" -> "run" , "better" -> "good" , "best" -> "good"
# the whole idea is to make the model focus on core meanings hence making them smarter. 

# the libraries that we are to use are 
# 1) numpy - efficient storage of tokenized data using numpy arrays.  It can also help in organzization of data by cleaning unnecessary words like "is" or "the" or even making sentences the same length. it can also count how many times a word appears in the text. 

# 2) tensorflow - helps in making it easy to process text and train machine learning models by:
# a) tokenizing the data using toools such as tokenizer from tensorflow.keras.preprocessing.text.Tokenizer or tensorflow.keras.layers.experimental.preprocessing.TextVectorization to break down sentences into smaller parts (TOKENS) and convert them into numers for the model. 
# b) handlin lemmatized data after reducing them to their base form using tools like NLTK or spaCy it can work with these cleaned data to make the model understand texts better. 
# c)it converts tokens into meaningfl number formats using layers like embedding. these embeddings helps models understand the relationships between words.
# d) it also ensure that all tokenized sentences have the same length by adding zeros where needed. 
# e) model training  - it uses this processed text to train models for tasks like sentiment analysis, text generation, and more.



import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

# the random in this case prvides tools for generating random numbers and shuffling words. it shuffles training data so that to ensure that the model does not learn in a fixed order and this ensures generalization. 
#JSON ensures reading, parsing and manipulation of JSON data. it is used to read the intents file.
#numpy provides the support with working with arrays and numerical computations efficiently. in the code below, it converts the the data into numPy arrays(train_x, train_y) that are required by tensorflow for neural network training. 
# tensorflow is used in neural network structure definition, training the models on the prepared data and saves the trained chatbot model. 
#nltk is a library that is used for natural language processing task and below, it is used to provide tools like word_tokenize for word tokenization and WordNetLemmatizer for lemmatization.
#WordNetLemmatizer is used to convert words to their base form.

# we will first create an instance of the WordNetLemmatizer class from the nltk.stem module.
lemmatizer = WordNetLemmatizer()

# we will then load the intents file that we created in the previous step.
#then we will convert it into a python object using the json.loads() function.
#this will give us a dictionary object that we can use to access the intents and their corresponding patterns and responses.
intent_file = json.loads(open('intents.json').read())

# remember when we tokenize the data, it will need to be stored somewhere and this is why we need this array.
words = []

# we will also need to store the classes that the intents belong to. this is because we will be using a neural network to classify the intents.
#These tags represent the categories the chatbot can classify user inputs into.
classes = []


#we will also need to store the intents and their corresponding tags. e.g.
#{
 # "tag": "greeting",
  #"patterns": ["Hello, how are you?", "Hi there!"]
#}
# documents will store:

# documents = [
 # (['Hello', 'how', 'are', 'you'], 'greeting'),
#  (['Hi', 'there'], 'greeting')
#]
#this is important as it helps classify each greeting to the correct tag.
documents = []


#we will also have to store ignored letters. these are characters that we do not want to consider when processing sentences. 
# we ignore them as the punctuation marks do not add meaning to the chatbot understanding and ignoring them helps focus on important words in the sentence.
#Example: For the sentence "Hello, how are you?", the program will:

#Tokenize it into: ['Hello', ',', 'how', 'are', 'you', '?'].
#Remove , and ?, resulting in: ['Hello', 'how', 'are', 'you'].
ignore_letters = ['!', '?', ',', '.']


# Iterating through each intent in the intents file.
for intent in intents['intents']:  
    # Iterating through the patterns in the intent.
    for pattern in intent['patterns']:  
        # We will use the word_tokenize function from the nltk library to tokenize the pattern.
        wordList = nltk.word_tokenize(pattern)
        # We now initialize the empty list we created earlier with the words in the pattern.
        words.extend(wordList)
        # Adding the word list and tag to the documents list as a tuple.
        documents.append((wordList, intent['tag']))
        # Adding the tag to the classes list if it is not already there.
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# We will now lemmatize the words and remove the words that are in the ignore_letters list.
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# We will then sort the words and remove duplicates, storing them back in the word list.
words = sorted(list(set(words)))
# This ensures that the words contain meaningful and unique words only.

# Example:
# Input: words = ['Hello', 'hello', '?', 'How', 'are', 'you']
# After processing: words = ['are', 'hello', 'how', 'you']

# Removes the duplicate intent tags and sorts the classes list.
classes = sorted(set(classes))

# Example:
# Input: classes = ['greeting', 'goodbye', 'greeting']
# After processing: classes = ['goodbye', 'greeting']

# The words and classes are then stored in the words.pkl and classes.pkl files respectively using the pickle library.
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# This is done so that the words and classes can be loaded later when training the model and not pass the model through the whole process again.

# Initialize the empty training list that holds the data that will be used to train the model. the training data consists of the tokenized words and the corresponding tag.
training = []
outputEmpty = [0] * len(classes) # this creates a list of zeros that has a length that is equal to the total number of tags that are in the class list.

#Example:
#If classes = ['goodbye', 'greeting'], then:
#outputEmpty = [0, 0]


# we will the iterate through each document in documents list. remember that documents list contains the tokenized words with their tags so each document object in documents contains a document[0] representing the words and the document[1] for the tag representing the intent. 
for document in documents:
    bag=[] #is an empty list that is used to store a list of 1s and 0s that represent the presence or absense of each word in the input pattern
    wordPatterns = document[0] #this is the tokenized words in the input pattern extracted and we want to compare them from the words list that we created earlier.
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns] #we converted them to lower case so that the model does not treat the same word in different cases as different words.
    #we then will lemmatize the word using the .lemmatize() so that we can reduce the words to their base form. this is important as it helps the model to understand the words better.


    # we then iterate through the words list and check if the word is in the wordPatterns. if it is, we append 1 to the bag list else we append 0.
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
        # If the word from words is found in wordPatterns (the list of words from the current document), it appends a 1 to the bag.
        #If the word from words is not found in wordPatterns, it appends a 0 to the bag.
        #This creates a "bag of words" representation, where each word from words is marked as present (1) or absent (0) in the current document's pattern.

        #
        outputRow = list(outputEmpty)
        outputRow[classes.index(document[1])] = 1




   








