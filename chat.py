import torch
from model import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  
#loading the file  
File = 'dataTFIDF.pth'
data = torch.load(File)


#loading all the data from the file
input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

#our previous model
from model import NeuralNet
model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()


#reading our json file will be required to print the output of our bot
import json
with open('intents.json','r') as f:
    intents = json.load(f)

#chatting using out trained model
from datapreprocessing import stemmingText,tokenize,bagOfWords
import numpy as np
import random

bot_name = "PokeMeNot"
print("Let's chat! (type 'quit' to exit)")
print(tags)
while True:
    #to take input
    sentence = input("You: ")
    if sentence == "quit":
        break 
    
    # if they want to chat
    # we use all the previous pre processing we have done in the data
    # tokenisation and removing punctutaion
    # then stemmin g the data
    # the bag of words
    sentence = tokenize(sentence)
    sentence = stemmingText(sentence)
    X = np.array(bagOfWords(sentence, all_words)) #numpy array
    X = X.reshape(1, X.shape[0]) #add a dimension
    X = torch.from_numpy(X)#converted it into a tensor


    #our output from our trained model 
    output = model.ffnn(X)
    #ouput is nothing but set of values for all tags in tensor

    #the one with the highest value
    _, predicted = torch.max(output, dim=1)#dim=1 says along row

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    #we converted that into probability
    
    prob = probs[0][predicted.item()]
    #the one with highest probability

    if prob.item() > 0.5:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")

