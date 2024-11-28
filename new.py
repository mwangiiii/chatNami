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
intents = json.loads(open('intents.json').read())

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

        #one hot encoding for the intent is performed by the below line:
        outputRow = list(outputEmpty)
        outputRow[classes.index(document[1])] = 1

        #one hot encoding is  a method of representing categorical data (in this case it is the intent tags) as binary vector and the length of this vector is equal to the number of unique categories (or classes) and each element of the vector is either 0 or 1. 
        #the one hot encoding is done so that the model can understand the data better.
        # 1 means that the intent is present in the current document and 0 means that it is not.
        # for example if you have  three intents: ["greeting", "goodbye", "weather"], a one-hot encoded vector for the "greeting" intent would be:

        #[1, 0, 0] (indicating that "greeting" is present and the others are not)

        #outputEmpty is a list of zeros that has a length equal to the number of classes (intent tags). For example, if you have three intents (["greeting", "goodbye", "weather"]), outputEmpty would be [0, 0, 0].
        #outputRow is a copy of this list that is initialized to zeros. It is then modified to have a 1 at the index corresponding to the current document's tag.
        # For example, if the current document's tag is "greeting" and the classes list is ["greeting", "goodbye", "weather"], outputRow would be [1, 0, 0].
        # This creates a one-hot encoded representation of the intent tag for the current document.
        # The bag and outputRow lists are then appended to the training list, which will be used to train the model.
        # The training list contains the bag of words representation of the input pattern and the one-hot encoded representation of the intent tag.
        # This data will be used to train the neural network model to classify user inputs into the correct intent tags.
        # The training list is then shuffled to ensure that the model does not learn in a fixed order.
        # This helps in generalization and prevents the model from overfitting to the training data.
        # The training list is then converted to a numpy array, which is required by TensorFlow for training neural network models.
        # The training data is then saved to the training_data.npy file using the np.save() function.
        # This file will be loaded later when training the model.

        #here the bag of words is combined with the outputRow to create training data
        #when you concatenate using the + character, it combines the two into a list that contains:
        #1) The bag of words that tells you which words were in the sentence. 
        #2) The outputRow that tells you which intent category the sentence belongs to.
        # So, for an example sentence "Hello, how are you?" with a tag of "greeting", you might get:
        #Bag of Words: [1, 1, 1, 1] (because "hello", "how", "are", and "you" are in the predefined words list).
        #One-Hot Encoding: [0, 1, 0] (assuming "greeting" is the second class in the list of possible intents).
# the combined data will be [1,1,1,1,0,1,0] and this will be stored in the training list.
        training.append(bag + outputRow)

#shuffle the training data so that the model does not learn in a fixed order.
random.shuffle(training)

# Convert the training data into a numpy array as it is required by TensorFlow for training neural network models.
# Converting them into numpy arrays is important as it also helps in slicing the data into trainX and trainY 
# where trainX contains the bag of words and trainY contains the one-hot encoded intent tags.

# Ensure that all elements in 'training' are homogeneous (same shape and type).
# This is done by iterating through 'training' and ensuring all sequences are padded or truncated to the same length.

# Calculate the required length based on words and classes.
required_length = len(words) + len(classes)

# Preprocess the training data to ensure homogeneity.
training = np.array([
    np.pad(t, (0, max(0, required_length - len(t))), mode='constant')[:required_length]
    for t in training
], dtype=np.float32)

# Split the training data into trainX and trainY where trainX contains the bag of words and trainY contains the one-hot encoded intent tags.
# This below line extracts the input features (trainX) from the training array.
# training[:, :len(words)] - slices the training array to get all rows (:) and the first len(words) columns (since words contains all the words in your vocabulary).
# This represents the "bag of words" (the presence or absence of words in the input patterns).
# Example: If words = ['hello', 'how', 'are', 'you'], and training contains a row like [1, 0, 1, 1, 0, 1],
# this line will extract the first four elements [1, 0, 1, 1] (which represent the presence of these words in the input pattern).
trainX = training[:, :len(words)]


#This below line extracts the output labels (trainY) from the training array.
#training[:, len(words):] - slices the training array to get all rows (:) and the columns starting from the index len(words) to the end.
#This represents the one-hot encoded intent tags.
#Example: If classes = ['greeting', 'goodbye'] and training contains a row like [1, 0, 1, 0], this line will extract the last two elements [1, 0] (which represent the intent tag "greeting").
trainY = training[:, len(words):]

#Example:
#Let’s imagine:

#words = ['hello', 'how', 'are', 'you']
#classes = ['greeting', 'goodbye', 'weather']
#A document's pattern is "hello how are you" with the tag "greeting".
#After processing this document, we might have:

#Bag of Words (trainX): [1, 1, 1, 1] (indicating all words are present)
#One-Hot Encoded Label (trainY): [1, 0, 0] (indicating the intent is "greeting")
#So, training might look like:

#[[1, 1, 1, 1, 1, 0, 0]]
#After running the code:

#trainX will be:

# [[1, 1, 1, 1]]
# trainY will be:
# python
# Copy code
# [[1, 0, 0]]
# This is the data that will be used to train the neural network model.

# the code below is for training a neural network model using the training data that we prepared earlier.


model = tf.keras.Sequential() # the line  creates a sequential model which is a linear stack of layers that are stacked onto each other hence creating a feedfoward neural network. This is the simple way of building a model.

#Purpose: This adds the first layer to the model, a Dense layer (fully connected layer).
#128 specifies the number of neurons in this layer.
#input_shape=(len(trainX[0]),) means that the model expects inputs where the shape of each input vector matches the length of trainX[0] (i.e., the number of words in your vocabulary).
#activation='relu' applies the ReLU (Rectified Linear Unit) activation function, which helps introduce non-linearity to the model.
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))


#Purpose: Adds a Dropout layer to prevent overfitting by randomly setting a fraction (0.5 in this case) of input units to 0 during training. This forces the model to learn more robust features by not relying too heavily on any one feature.
#0.5 means 50% of the neurons in this layer will be randomly dropped during training.
model.add(tf.keras.layers.Dropout(0.5))


model.add(tf.keras.layers.Dense(64, activation = 'relu'))# Adds a second Dense layer with 64 neurons. Again, it uses the ReLU activation function.


model.add(tf.keras.layers.Dropout(0.5)) # Adds another Dropout layer to prevent overfitting.


# Adds the output layer, which is another Dense layer.
# len(trainY[0]) specifies the number of output neurons, which corresponds to the number of unique intents or classes in your dataset (each class is a different intent).
# activation='softmax' applies the Softmax activation function, which converts the output values into probabilities. It outputs a probability distribution over the intents.
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Compiling the model:
# Creates a Stochastic Gradient Descent (SGD) optimizer.
# learning_rate=0.01: The learning rate controls how much to change the model weights during each update. A smaller learning rate could make training slow, and a larger one could make it unstable.
# momentum=0.9: Momentum helps accelerate gradients vectors in the right directions, thus leading to faster converging.
# nesterov=True: Nesterov momentum helps improve the optimizer's performance by anticipating the future gradient direction.
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Purpose: Configures the model for training.
# loss='categorical_crossentropy': This is the loss function used for multi-class classification problems. It measures how well the predicted probabilities match the actual one-hot encoded labels.
# optimizer=sgd: Specifies the optimizer created earlier.
# metrics=['accuracy']: The model will evaluate its accuracy during training.
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# This trains the model on the data.
# np.array(trainX): The training input data (bag of words).
# np.array(trainY): The training target data (one-hot encoded intents).
# epochs=200: The number of times the model will see the entire training data. 200 epochs means the model will train on the data 200 times.
# batch_size=5: The number of training samples that will be passed through the model at a time. A batch size of 5 means that for each step, 5 samples are processed before the weights are updated.
# verbose=1: Prints out the progress of the training (it will display the progress bar and other details).
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# Saves the trained model to a file called chatbot_model.h5.
# This allows you to load the trained model later without retraining it.
# The hist object contains the training history (such as loss and accuracy over epochs) and can be saved with the model, although this is not very common for just saving the model itself.
model.save('chatbot_trained_model_file.h5')  # Removed hist argument, as it's not valid for save()

# Prints "Done" to indicate that the model has been trained and saved successfully.
print('Done')









   








