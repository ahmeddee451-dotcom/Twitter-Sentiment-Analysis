from flask import Flask,request,jsonify
import re 
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk

#initiallize the Flask

app = Flask(__name__)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# load the .h5 file

model = tf.keras.models.load_model("sentimnent_rnn_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

    max_lenth = 100


    def claen_text(text):
        text = re.sub(r'@[\w]*','',text)
        text = re.sub(r'#[\w]*','',text)
        text = re.sub(r'[^a-zA-Z\s]','',text)
        text = text.lower()
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)

def preprocess(text):
    cleaned = claen_text(text)
    sq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sq,maxlen= max_lenth)
    return padded

@app.route("/predict",methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', "")
        if text.strip() == "":
            return jsonify({'error': 'text is required'}),400
        processed = preprocess(text)
        prediction = model.predict(processed)[0][0]
        sentiment = 'positive' if prediction>=0.5 else 'negative'
        return jsonify({'sentiment': sentiment,
                        'confidence': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}),500
@app.route('/',methods=['GET'])
def home():
    return jsonify({'message': "sentiment API is running!"})

if __name__ =='__main__':
    app.run(debug=False)

    
    

