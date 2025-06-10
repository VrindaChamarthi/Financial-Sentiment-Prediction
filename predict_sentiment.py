from flask import Flask, request, render_template
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from googletrans import Translator
import os

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
custom_objects = {'TFBertModel': TFBertModel}

MODEL_PATH = "/Users/vrinda/PycharmProjects/MAJOR PROJECT/sentiment_model.h5"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)

# Initialize translator
translator = Translator()

# Flask app
app = Flask(__name__)

# Prediction function with translation
def predict_sentiment(statement, model, tokenizer, max_length=128):
    # Translate to English
    translated = translator.translate(statement, dest='en').text

    # Tokenize
    inputs = tokenizer(
        [translated],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )

    # Predict
    predictions = model.predict([inputs["input_ids"], inputs["attention_mask"]])
    predicted_label = tf.argmax(predictions, axis=1).numpy()[0]
    label_map = {
        0: "Company's stock price will reduce.",
        1: "Company's stock price remains unaffected.",
        2: "Company's stock price will increase."
    }
    return label_map[predicted_label], translated

# Route for homepage
@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    headline = ""
    translated_text = ""
    if request.method == "POST":
        headline = request.form["headline"]
        sentiment, translated_text = predict_sentiment(headline, model, tokenizer)
    return render_template("index.html", sentiment=sentiment, headline=headline, translated=translated_text)

# Run app
if __name__ == "__main__":
    app.run(debug=True)























#from flask import Flask, request, render_template
# import tensorflow as tf
# from transformers import BertTokenizer, TFBertModel
# import os
#
# # Load tokenizer and model
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# custom_objects = {'TFBertModel': TFBertModel}
#
# MODEL_PATH = "/Users/vrinda/PycharmProjects/MAJOR PROJECT/sentiment_model.h5"
# model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
#
# # Flask app
# app = Flask(__name__)
#
# # Prediction function
# def predict_sentiment(statement, model, tokenizer, max_length=128):
#     inputs = tokenizer(
#         [statement],
#         max_length=max_length,
#         padding="max_length",
#         truncation=True,
#         return_tensors="tf"
#     )
#
#     predictions = model.predict([inputs["input_ids"], inputs["attention_mask"]])
#     predicted_label = tf.argmax(predictions, axis=1).numpy()[0]
#     label_map = {0: "Company's stock price will reduce.", 1: "Company's stock price remains unaffected.", 2: "Company's stock price will increase."}
#     return label_map[predicted_label]
#
# # Route for homepage
# @app.route("/", methods=["GET", "POST"])
# def home():
#     sentiment = None
#     headline = ""
#     if request.method == "POST":
#         headline = request.form["headline"]
#         sentiment = predict_sentiment(headline, model, tokenizer)
#     return render_template("index.html", sentiment=sentiment, headline=headline)
#
# # Run app
# if __name__ == "__main__":
#     app.run(debug=True)
#
#
#
#
