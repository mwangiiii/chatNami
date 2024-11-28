# chatNami

Welcome to the **Chatbot Project**, a conversational AI model built using **Python**, **TensorFlow**, and **NLTK**. This chatbot can process user input, predict intents, and provide relevant responses. It is trained on intents and patterns provided in a JSON file.

## 🚀 Features

- **Intent Classification:** Accurately predicts the user's intent.
- **Custom Responses:** Provides appropriate replies based on the trained data.
- **Extensibility:** Easily extendable with new intents and patterns.
- **Interactive Conversations:** Engages in dynamic conversations with users.

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow 2.x**
- **NLTK (Natural Language Toolkit)**
- **NumPy**
- **JSON for intent definitions**
- **Pickle for serialization**

---

## 📂 Project Structure

```plaintext
.
├── chatbot.py               # Main chatbot script
├── training.py              # Script for training the chatbot
├── intents.json             # Intent patterns and responses
├── chatbot_trained_model.h5 # Trained model file
├── words.pkl                # Serialized vocabulary
├── classes.pkl              # Serialized classes
└── README.md                # Project documentation

## 🧠 How It Works

1. **Tokenization & Lemmatization:**  
   - User input is tokenized and lemmatized for preprocessing.

2. **Bag of Words (BoW):**  
   - The input sentence is converted into a bag-of-words representation.

3. **Model Prediction:**  
   - The pre-trained TensorFlow model predicts the intent from the processed input.

4. **Response Generation:**  
   - Based on the predicted intent, a random response is fetched from the `intents.json`.

---

## 🛑 Prerequisites

1. Install **Python 3.8** or higher.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## ⚙️ Installation and Setup

### Clone the repository:
```bash
git clone https://github.com/your-username/chatbot.git
cd chatbot
```

### Install dependencies:

```
pip install -r requirements.txt
```

### Train the chatbot (if not already trained):

```
python new.py
```

### Run the chatbot:

```
python chatbot.py
```
## 🧪 Training the Chatbot

> Modify intents.json to include new intents, patterns, and responses.

Run the training script:

```
python training.py
```

> This will generate the following files:

1.    chatbot_trained_model.h5 (Trained model)
2.    words.pkl (Vocabulary)
3.    classes.pkl (Intent classes)

## 🔧 Customization

### Adding New Intents

Open intents.json.

Add a new intent with the following format:

```
{
    "tag": "new_tag",
    "patterns": [
        "Example pattern 1",
        "Example pattern 2"
    ],
    "responses": [
        "Example response 1",
        "Example response 2"
    ],
    "context": [""]
}
```

Retrain the model:

```
python training.py
```

## 🛠️ Debugging

> If the chatbot fails to understand your inputs:

1.    Verify the intents.json file for proper structure.
2.    Check the training script for errors.
3.    Add debugging statements in chatbot.py to inspect the Bag of Words and predictions.


