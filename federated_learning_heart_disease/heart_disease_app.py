import numpy as np
import pickle
from flask import Flask, request, render_template
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import tensorflow as tf
tf.__version__
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import syft as sy
import pickle
import torch as th
hook = sy.TorchHook(th)
from torch import nn, optim

input_size = 13
learning_rate = 0.001
num_iterations = 5000

class LogisticRegression(torch.nn.Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def decide(y):
    return 1. if y >= 0.5 else 0.

decide_vectorized = np.vectorize(decide)

to_percent = lambda x: '{:.2f}%'.format(x)

# Specify a path
filename = "federated_model.sav"
model = pickle.load(open(filename, 'rb'))

def decide(y):
    return 1. if y >= 0.5 else 0.

decide_vectorized = np.vectorize(decide)

to_percent = lambda x: '{:.2f}%'.format(x)

def compute_accuracy(model, input):
    prediction = model.predict(input)
    
    return prediction


def get_input_and_output(data):
    input = Variable(torch.tensor(data[:, :], dtype = torch.float32))
    return input
# Create application
app = Flask(__name__)

# Bind home function to URL


@app.route('/')
def home(): 
    return render_template('index.html')

# Bind predict function to URL


@app.route('/showclf')
def showclf():
    return render_template('logistic_classifier.html')




@app.route('/predictclf', methods=['POST'])
def predictclf():

    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    col = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_data1=pd.DataFrame([features],columns = col)
    input_data = np.array(input_data1,dtype='float32')
    test_input = get_input_and_output(input_data)
    prediction = compute_accuracy(model,test_input)

    print(prediction)
    output = prediction
    print(output)

    # Check the output values and retrive the result with html tag based on the value
    if output == 1.:
        return render_template('logistic_classifier.html',res2='HEART DISEASE')
    else:
        return render_template('logistic_classifier.html',res2='NO HEART DISEASE')


if __name__ == '__main__':
    # Run the application
    app.run(debug=True)



