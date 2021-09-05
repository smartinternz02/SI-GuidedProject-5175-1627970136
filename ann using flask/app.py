import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from tensorflow.keras.models import load_model
app = Flask(__name__)

model = load_model("fish.h5",compile=False)
sc=pickle.load(open("fishes.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    z = request.form['weight']
    a = request.form['height']
    b = request.form['length1']
    c = request.form['length2']
    d = request.form['length3']
    e = request.form['width']
    total = [[z,a,b,c,d,e]]
    pred= model.predict(sc.transform(total))
    species = ["Bream","Parkki","Perch","pike","roach","smelt","whitefish"]
    prediction = species[np.argmax(pred)]
    
    if(prediction=='Bream'):
        output = "The species is bream"
    
    elif(prediction=='parkki'):
        output = "The species is parkki"
        
    elif(prediction=='perch'):
        output = "The species is perch"
        
    elif(prediction=='pike'):
        output = "The species is pike"
        
    elif(prediction=='roach'):
        output = "The species is roach"
        output = "The species is roach"
    
    elif(prediction=='smelt'):
        output = "The species is smelt"
    
    elif(prediction=='whitefish'):
        output = "The species is whitefish"
    else:
        output = "Species not found"


    return render_template('index.html', prediction_text='Result: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
