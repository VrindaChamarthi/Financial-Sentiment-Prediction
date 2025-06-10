import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel


def load_test_data(csv_path, max_length=128):
    df = pd.read_csv(csv_path, encoding='latin1', names=["Sentiment", "News Headline"], header=None)
    df["News Headline"] = df["News Headline"].astype(str).fillna("")
    df["Sentiment"] = df["Sentiment"].astype(str).str.lower().str.strip()
    sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["Sentiment"].map(sentiment_mapping)
    df = df.dropna(subset=["label"])

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokenized = tokenizer(
        list(df["News Headline"]),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )

    input_ids = tokenized["input_ids"]
    attention_masks = tokenized["attention_mask"]
    labels = df["label"].values

    _, test_ids, _, test_masks, _, test_labels = train_test_split(
        input_ids, attention_masks, labels, test_size=0.2, random_state=42
    )

    return test_ids, test_masks, test_labels


def rebuild_model(max_length=128):
    input_ids_in = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask_in = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    bert_output = bert_model(input_ids_in, attention_mask=attention_mask_in)[0]

    conv = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(bert_output)
    pool = tf.keras.layers.GlobalMaxPooling1D()(conv)
    drop = tf.keras.layers.Dropout(0.3)(pool)
    output = tf.keras.layers.Dense(3, activation='softmax')(drop)

    model = tf.keras.Model(inputs=[input_ids_in, attention_mask_in], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def evaluate_model():
    csv_path = "/Users/vrinda/PycharmProjects/MAJOR PROJECT /all-data.csv"
    weights_path = "/Users/vrinda/PycharmProjects/MAJOR PROJECT/sentiment_model.h5"
    max_length = 128

    print("📦 Loading test data...")
    test_ids, test_masks, test_labels = load_test_data(csv_path, max_length)

    print("🔁 Rebuilding model architecture...")
    model = rebuild_model(max_length)

    print("📥 Loading trained weights...")
    model.load_weights(weights_path)

    print("🧠 Making predictions...")
    predictions = model.predict([test_ids, test_masks])
    predicted_labels = np.argmax(predictions, axis=1)

    print("\n📊 Classification Report:")
    print(classification_report(test_labels, predicted_labels, target_names=["negative", "neutral", "positive"]))

    print("\n🧮 Confusion Matrix:")
    print(confusion_matrix(test_labels, predicted_labels))


if __name__ == "__main__":
    evaluate_model()
