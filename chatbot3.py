# Importing the necessary libraries
import nltk
import numpy as np
import random
import string  # to process standard python strings
import os
from flask import Flask, render_template, request
import nexmo

# Generating Response
# To generate a response from our bot for input questions, the concept of document similarity will be used. So we begin by importing the necessary modules.
# From scikit learn library, import the TFidf vectorizer to convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer
# Also, import cosine similarity module from scikit learn library
from sklearn.metrics.pairwise import cosine_similarity

f = open(r'D:\flask-class\data.txt', errors='ignore')
client = nexmo.Client(key='b5662319', secret='I2YQJ1Lv2FdqpfaC')
# Corpus
# For our example, we will be using the Wikipedia page for chatbots as our corpus. Copy the contents from the page and place it in a text file named ‘chatbot.txt’. However, you can use any corpus of your choice.
# Reading in the data
# We will read in the corpus.txt file and convert the entire corpus into a list of sentences and a list of words for further pre-processing.
raw = f.read()
raw = raw.lower()  # converts to lowercase
nltk.download('punkt')  # first-time use only
nltk.download('wordnet')  # first-time use only
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words
# Let see an example of the sent_tokens and the word_tokens
# sent_tokens[:2]
# [
#    'a chatbot (also known as a talkbot, chatterbot, bot, im bot, interactive agent, or artificial conversational entity) is a computer program or an artificial intelligence which conducts a conversation via auditory or textual methods.',
#    'such programs are often designed to convincingly simulate how a human would behave as a conversational partner, thereby passing the turing test.']
# word_tokens[:2]
# ['a', 'chatbot', '(', 'also', 'known']
# Pre-processing the raw text
# We shall now define a function called LemTokens which will take as input the tokens and return normalized tokens.
lemmer = nltk.stem.WordNetLemmatizer()

# WordNet is a semantically-oriented dictionary of English included in NLTK.


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

    # Keyword matching
    # Next, we shall define a function for a greeting by the bot i.e if a user’s input is a greeting, the bot shall return a greeting response.ELIZA uses a simple keyword matching for greetings. We will utilize the same concept here.
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there",
                      "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/menu', methods=['POST', 'GET'])
def menu():
    return render_template('menu.html')


@app.route('/instruct', methods=['POST', 'GET'])
def instruct():
    return render_template('instruct.html')


@app.route('/send', methods=['POST'])
def send():
    user_message = request.form['user_message']
    response = client.send_message(
        {'from': 'Nexmo', 'to': 919167959537, 'text': user_message})
    response_text = response['messages'][0]
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    flag = True
    while flag == True:
        user_response = request.form['user_response']
        user_response = user_response.lower()
        if (greeting(user_response) != None):
            robo_response = greeting(user_response)
            return render_template('index.html', user_response=user_response, robo_response=robo_response)
        else:
            robo_response = response(user_response)
            sent_tokens.remove(user_response)
            return render_template('index.html', user_response=user_response, robo_response=robo_response)


@app.route('/card')
def card():
    return render_template('card.html')


@app.route('/cash')
def cash():
    return render_template('cash.html')


@app.route('/process1', methods=['POST', 'GET'])
def process1():
    item1 = 0
    qty1 = 0
    item2 = 0
    qty2 = 0
    item3 = 0
    qty3 = 0
    grandTotal = 0

    starter = request.form['starter']
    if starter == 'None':
        item1 = 0
    if starter == 'Cheese Balls':
        item1 = 30
    elif starter == 'Samosas':
        item1 = 20
    elif starter == 'Wings':
        item1 = 40

    starter_qty = request.form['starter_qty']
    if starter_qty == '1':
        qty1 = 1
    elif starter_qty == '2':
        qty1 = 2
    elif starter_qty == '3':
        qty1 = 3
    elif starter_qty == '4':
        qty1 = 4
    elif starter_qty == '5':
        qty1 = 5

    main_course = request.form['main_course']
    if main_course == 'None':
        item2 = 0
    if main_course == 'Chicken Biryani':
        item2 = 250
    elif main_course == 'Schezwan rice':
        item2 = 220
    elif main_course == 'Pizza':
        item2 = 300

    main_course_qty = request.form['main_course_qty']
    if main_course_qty == '1':
        qty2 = 1
    elif main_course_qty == '2':
        qty2 = 2
    elif main_course_qty == '3':
        qty2 = 3
    elif main_course_qty == '4':
        qty2 = 4
    elif main_course_qty == '5':
        qty2 = 5

    dessert = request.form['dessert']
    if dessert == 'None':
        item3 = 0
    if dessert == 'Chocolate shake':
        item3 = 120
    elif dessert == 'Ice cream':
        item3 = 80
    elif dessert == 'Gulab Jamun':
        item3 = 30

    dessert_qty = request.form['dessert_qty']
    if dessert_qty == '1':
        qty3 = 1
    elif dessert_qty == '2':
        qty3 = 2
    elif dessert_qty == '3':
        qty3 = 3
    elif dessert_qty == '4':
        qty3 = 4
    elif dessert_qty == '5':
        qty3 = 5

    grandTotal = item1*qty1 + item2*qty2 + item3*qty3

    return render_template('index.html', grandTotal=grandTotal)


if __name__ == "__main__":
    app.run(debug=True)
