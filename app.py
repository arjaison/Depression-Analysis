import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modelfinchange.pkl', 'rb'))

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
    #print("Helooo     ",int_features)
    
    final_features = np.array(int_features)
    #print(type(final_features))
    #final_features=final_features.reshape(1,-1)  
    final_features=np.delete(final_features,(0,2,7,8,10,12,22))
    final_features=[int(i) for i in final_features] 
    
    
    #print("Byeeeeee     ",final_features)
    #print(type(final_features))
    #final_features=final_features.delete['What is your name?','Which school are you studying in?','What is your favourite food?','How do you workout everyday?','How do you start your day?','How do you usually feel after a nap?', 'Mention a health problem which troubles you often']
    prediction = model.predict([final_features])
    #print(type(prediction))
    print(prediction)
    output = prediction

    return render_template('index.html', prediction_text='Result: $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)