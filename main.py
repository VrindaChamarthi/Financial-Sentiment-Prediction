import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
from keras.callbacks import EarlyStopping



def load_and_preprocess_data(csv_path, max_length=128):
    # Load dataset
    # df = pd.read_csv("/Users/vrinda/PycharmProjects/MAJOR PROJECT /all-data.csv")
    df = pd.read_csv("/Users/vrinda/PycharmProjects/MAJOR PROJECT /all-data.csv", encoding='latin1', names=["Sentiment", "News Headline"],
                     header=None)
    # cols are string, no nulls, lowercase, no extra spaces before and after
    df["News Headline"] = df["News Headline"].astype(str).fillna("")
    df["Sentiment"] = df["Sentiment"].astype(str).str.lower().str.strip()

    # mapping lables to numeric vals
    sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    df["label"] = df["Sentiment"].map(sentiment_mapping)

    # check for null vals
    df = df.dropna(subset=["label"])

    # tokeniser is loaded for tokenisisng input
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_texts(texts, max_length):
        texts = [str(t) for t in texts]
        return tokenizer(
            texts,
            max_length=max_length, #length of each token
            padding='max_length',
            truncation=True, #if len greater, truncate it to get it to max len
            return_tensors='tf'
        )

    tokenized_data = tokenize_texts(df["News Headline"], max_length) #tokenise input vals


    input_ids_all = tokenized_data["input_ids"].numpy() # tokenised input ids converted to array
    attention_masks_all = tokenized_data["attention_mask"].numpy() #attention mask is to find out which is legit data and which is just padding (0s)
    labels_all = df["label"].astype(int).values #labels are int not string

    return input_ids_all, attention_masks_all, labels_all


def split_data(input_ids_all, attention_masks_all, labels_all):
    train_ids, test_ids, train_masks, test_masks, train_labels, test_labels = train_test_split(
        input_ids_all, #use the numpy arrays
        attention_masks_all,
        labels_all,
        test_size=0.2,
        random_state=42
    )
    print("Train/Test split completed.")
    return train_ids, test_ids, train_masks, test_masks, train_labels, test_labels


def build_model(max_length=128):
    input_ids_in = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids") #input for bert
    attention_mask_in = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask") #atentioin mask input for bert

    bert_model = TFBertModel.from_pretrained("bert-base-uncased") # use the uncased bert model
    bert_output = bert_model(input_ids_in, attention_mask=attention_mask_in)[0] #each toekn is represented as a hidden_size vector for complete understadning

    conv = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(bert_output) #3 tokens at a time passed through 128 filters
    pool = tf.keras.layers.GlobalMaxPooling1D()(conv) #the max val from each flter is selected to get 128 features
    drop = tf.keras.layers.Dropout(0.3)(pool) #randomly drop zeroes to prevent overfitting
    output = tf.keras.layers.Dense(3, activation='softmax')(drop) #use softmax to classify into 3 classes and give 0.67 type values from the 128 features

    model = tf.keras.Model(inputs=[input_ids_in, attention_mask_in], outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), # 2 * 10^-5 (small learning weights to not disturb the pre trained weights of the bert model)
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


def main():
    csv_path = "/Users/vrinda/PycharmProjects/MAJOR PROJECT /all-data.csv"
    max_length = 128

    input_ids_all, attention_masks_all, labels_all = load_and_preprocess_data(csv_path, max_length)
    train_ids, test_ids, train_masks, test_masks, train_labels, test_labels = split_data(
        input_ids_all, attention_masks_all, labels_all
    )

    model = build_model(max_length)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2, #if for 2 epochs gthere is no impovemnet, it stops (prevents overfitting)
        restore_best_weights=True
    )

    history = model.fit(
        [train_ids, train_masks],
        train_labels,
        validation_split=0.1,
        epochs=10,
        batch_size=16,
        callbacks=[early_stopping]
    )

    # Save trained model
    model.save("/Users/vrinda/PycharmProjects/MAJOR PROJECT/sentiment_model.h5")
    print("Model saved successfully.")

    loss, accuracy = model.evaluate([test_ids, test_masks], test_labels)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)



if __name__ == '__main__':
    main()






































# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from transformers import BertTokenizer, TFBertModel
#
#
# def load_and_preprocess_data(csv_path, max_length=128):
#     # Load your dataset (adjust the path/filename as needed)
#     df = pd.read_csv("/Users/vrinda/PycharmProjects/MAJOR PROJECT /fsa/financial_sentiment_dataset.csv")
#
#     # Ensure "text" is string type and fill any missing values
#     df["text"] = df["text"].astype(str)
#     df["text"] = df["text"].fillna("")
#
#     # Map sentiment values -1.0, 0.0, 1.0 to integer labels 0,1,2
#     sentiment_mapping = {-1.0: 0, 0.0: 1, 1.0: 2}
#     df["label"] = df["sentiment"].map(sentiment_mapping)
#
#     # Load BERT tokenizer (using the uncased model)
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#
#     # Function to tokenize texts
#     def tokenize_texts(texts, max_length):
#         # Force convert every element to a string
#         texts = [str(t) for t in texts]
#         return tokenizer(
#             texts,
#             max_length=max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='tf'
#         )
#
#     # Tokenize the text column
#     tokenized_data = tokenize_texts(df["text"], max_length)
#
#     # Convert TensorFlow tensors to NumPy arrays
#     input_ids_all = tokenized_data["input_ids"].numpy()
#     attention_masks_all = tokenized_data["attention_mask"].numpy()
#     labels_all = df["label"].values  # already numeric
#     labels_all = np.array(labels_all)
#
#     return input_ids_all, attention_masks_all, labels_all
#
#
# def split_data(input_ids_all, attention_masks_all, labels_all):
#     # Split data into training and testing sets (80/20 split)
#     train_ids, test_ids, train_masks, test_masks, train_labels, test_labels = train_test_split(
#         input_ids_all,
#         attention_masks_all,
#         labels_all,
#         test_size=0.2,
#         random_state=42
#     )
#     print("Train/Test split completed.")
#     print("Train shapes:", train_ids.shape, train_masks.shape, train_labels.shape)
#     print("Test shapes:", test_ids.shape, test_masks.shape, test_labels.shape)
#     return train_ids, test_ids, train_masks, test_masks, train_labels, test_labels
#
#
# def build_model(max_length=128):
#     # Define input layers for token ids and attention masks
#     input_ids_in = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
#     attention_mask_in = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
#
#     # Load pre-trained BERT model for embeddings
#     bert_model = TFBertModel.from_pretrained("bert-base-uncased")
#     # Get the last hidden state from BERT (shape: [batch, seq_len, hidden_size])
#     bert_output = bert_model(input_ids_in, attention_mask=attention_mask_in)[0]
#
#     # Apply a 1D convolution over the BERT embeddings
#     conv = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(bert_output)
#     # Global max pooling to capture the most salient features
#     pool = tf.keras.layers.GlobalMaxPooling1D()(conv)
#     # Optional dropout for regularization
#     drop = tf.keras.layers.Dropout(0.3)(pool)
#     # Final Dense layer for classification into 3 classes (negative, neutral, positive)
#     output = tf.keras.layers.Dense(3, activation='softmax')(drop)
#
#     # Construct the model
#     model = tf.keras.Model(inputs=[input_ids_in, attention_mask_in], outputs=output)
#
#     # Compile the model
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     model.summary()
#     return model
#
#
# def main():
#     # Adjust the CSV path if needed
#     csv_path = "/Users/vrinda/PycharmProjects/MAJOR PROJECT /fsa/financial_sentiment_dataset.csv"
#     max_length = 128
#
#     # Load and preprocess the data
#     input_ids_all, attention_masks_all, labels_all = load_and_preprocess_data(csv_path, max_length)
#
#     # Split the data
#     train_ids, test_ids, train_masks, test_masks, train_labels, test_labels = split_data(
#         input_ids_all, attention_masks_all, labels_all
#     )
#
#     # Build the model
#     model = build_model(max_length)
#
#     # -----------------------------
#     # 3. Training the Model
#     # -----------------------------
#     history = model.fit(
#         [train_ids, train_masks],
#         train_labels,
#         validation_split=0.1,
#         epochs=3,
#         batch_size=16
#     )
#
#     # ✅ Save the model after training
#     model.save("/Users/vrinda/PycharmProjects/MAJOR PROJECT /fsa/sentiment_model.h5")
#     print("✅ Model saved successfully.")
#
#     # -----------------------------
#     # 4. Evaluation
#     # -----------------------------
#     loss, accuracy = model.evaluate([test_ids, test_masks], test_labels)
#     print("Test loss:", loss)
#     print("Test accuracy:", accuracy)
#
#
# if __name__ == '__main__':
#     main()
