import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modelfinal.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    #print(type(int_features))
   
    
    final_features = np.array(int_features)
   
    final_features=np.delete(final_features,(0,2,7,8,10,12,22))
    final_features=[int(i) for i in final_features] 

    prediction = model.predict([final_features])
    
    print(prediction)
    output = prediction

    return render_template('index.html', prediction_text='Result: $ {}'.format(output))


if __name__ == "__main__":
   
    app.run(debug=True)