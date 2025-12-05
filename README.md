

#  Twitter Sentiment Analysis using RNN/LSTM

This project performs sentiment classification on tweet text to determine whether a comment is **Positive** or **Negative**. It uses Deep Learning (RNN & LSTM architectures), along with Flask and Streamlit for deployment.

---

##  Key Features

* Text cleanup, preprocessing, and tokenization
* Model training using SimpleRNN (and optional LSTM)
* Saving trained model and tokenizer
* Real-time API prediction using Flask
* Interactive UI using Streamlit

---

##  Tech Stack

* Python
* Pandas, NumPy
* TensorFlow / Keras
* NLTK (stopwords)
* Flask REST API
* Streamlit UI

---

##  Project Structure

```
├── tweets_data.csv
├── Sentiment_Analysis_Using_LSTM and RNN.py
├── flask_app.py
├── streamlit.py
├── sentimnent_rnn_model.h5
├── sentimnent_lstm_model.h5
└── tokenizer.pkl
```

---

##  Model Training

Training file processes the CSV dataset, cleans tweets, pads sequences, trains the model, and saves artifacts.

Model summary:

* Embedding layer
* Multiple SimpleRNN layers + Dropout
* Dense(sigmoid) output → binary classification

Artifacts saved:


sentimnent_rnn_model.h5
tokenizer.pkl
```

---

##  API Prediction (Flask)

Start the server:

```bash
python flask_app.py
```

### Send request:

```json
POST /predict
{
  "text": "This movie was great!"
}
```

### Response format:

```json
{
  "sentiment": "positive",
  "confidence": 0.92
}
```

---

##  Streamlit Interface

Run:

```bash
streamlit run streamlit.py
```

You can:

* Enter a tweet text
* Get sentiment result & probability score

---

##  Data Preprocessing Includes

✔ Removing hashtags
✔ Removing mentions
✔ Removing special characters
✔ Converting to lowercase
✔ Stopword removal
✔ Tokenizing + padding

---

##  Possible Improvements

* Add neutral class
* Add model evaluation metrics
* Deploy using Docker / Cloud hosting
* Use pre-trained embeddings (GloVe, Word2Vec)




