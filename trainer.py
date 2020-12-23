from flask import Flask, render_template, request
from chatterbot import ChatBot
chatbot = ChatBot("Robo")
from chatterbot.trainers import ChatterBotCorpusTrainer
import os

trainer = ChatterBotCorpusTrainer(chatbot)

trainer.train(
    
    "./data.yml"
)

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/process',methods=['POST'])
def process():
    user_input = request.form['user_input']
    bot_response=chatbot.get_response(user_input)
    bot_response=str(bot_response)
    print("Robo: "+bot_response)
    return render_template('index.html',user_input=user_input,bot_response=bot_response)

if __name__=="__main__":
	app.run(debug=True)
    



