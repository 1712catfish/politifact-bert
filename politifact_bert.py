import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import torch
import re
import nltk

from tensorflow.keras.layers import Dense, Input, Conv2D, LSTM, Bidirectional, Flatten, MaxPool2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from transformers import BertTokenizer, TFBertModel
from gensim.utils import simple_preprocess
from math import ceil


class BigClass:
    def __init__(self,
                 dataset_raw_train: pd.DataFrame,
                 dataset_raw_test: pd.DataFrame,
                 save_path: str,
                 labels: list[str] = ['FAKE', 'REAL']):
        self.raw_train_dataset = dataset_raw_train
        self.raw_test_dataset = dataset_raw_test
        self.save_path = save_path

        self.bert_tokenizer = None
        self.bert_model = None
        self.train_history = {}

        self.bert_dense_model = None
        self.bert_lstm_model = None
        self.bert_bilstm_model = None
        self.bert_cnn_model = None

        self.MAX_SEQ_LENGTH = 512
        self.EPOCHS = 10
        self.MODEL_TYPE = ['dense', 'lstm', 'bilstm', 'cnn']
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.LABELS = labels
        self.model_name = "bert-base-uncased"
        self.embed_len = 768

        nltk.download("stopwords")
        self.stop_words = set(nltk.corpus.stopwords.words("english"))

    def prepare_dataset(self):
        print("Training Dataset - ", self.raw_train_dataset.iloc[:, 1].value_counts())
        self.X_train_raw = self.raw_train_dataset.iloc[:, 0].tolist()
        y_train_raw = self.raw_train_dataset.iloc[:, 1].replace(self.LABELS, [0, 1]).tolist()

        self.y_train_raw = [[0, 0] for _ in range(len(y_train_raw))]
        for i, j in enumerate(y_train_raw):
            self.y_train_raw[i][j] = 1
        self.y_train_raw = np.array(self.y_train_raw, dtype=np.float32)

        print("Train data shape - ", self.raw_train_dataset.shape)

        print("Test Dataset - ", self.raw_test_dataset.iloc[:, 1].value_counts())
        self.X_test_raw = self.raw_test_dataset.iloc[:, 0].tolist()
        y_test_raw = self.raw_test_dataset.iloc[:, 1].replace(self.LABELS, [0, 1]).tolist()

        self.y_test_raw = [[0, 0] for _ in range(len(y_test_raw))]
        for i, j in enumerate(y_test_raw):
            self.y_test_raw[i][j] = 1
        self.y_test_raw = np.array(self.y_test_raw, dtype=np.float32)

        print("Test data shape - ", self.raw_test_dataset.shape)

    def prepare_bert_tokens(self):
        if not self.bert_model or not self.bert_tokenizer:
            self.bert_tokenizer, self.bert_model = self.load_bert_model()
            print("Loaded bert Tokenizer and Model.")
        else:
            print("bert Model and Tokenizer already loaded.")

        self.X_train_bert_tokens = self.get_bert_tokens(
            self.X_train_raw
        )

        self.X_test_bert_tokens = self.get_bert_tokens(
            self.X_test_raw
        )

        print("bert Tokens prepared.")

    def load_bert_model(self):
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = TFBertModel.from_pretrained(self.model_name)
        model.trainable = False

        return tokenizer, model

    def save_bert_tokens(self):
        os.makedirs(self.save_path + "/Tokens/bert/", exist_ok=True)

        np.savez(self.save_path + "/Tokens/bert/tokens.npz",
                 train_bert=self.X_train_bert_tokens,
                 test_bert=self.X_test_bert_tokens)

        print("Embeddings saved at path - {}".format(self.save_path + "/Tokens/bert/tokens.npz"))

    def load_bert_tokens(self, path=None):
        if path is None:
            numpy_file = np.load(self.save_path + "/Tokens/bert/tokens.npz")
        else:
            numpy_file = np.load(path)

        self.X_train_bert_tokens = numpy_file['train_bert']
        self.X_test_bert_tokens = numpy_file['test_bert']

        print("bert Embeddings successfully loaded.")

    def save_labels(self):
        os.makedirs(self.save_path + "/Tokens/bert-Labels/", exist_ok=True)
        np.savez(self.save_path + "/Tokens/bert-Labels/label.npz",
                 train=self.y_train_raw, test=self.y_test_raw)

        print("Labels saved at path - {}".format(self.save_path + "/Tokens/bert-Labels/label.npz"))

    def load_labels(self, path=None):
        if path is None:
            numpy_file = np.load(self.save_path + "/Tokens/bert-Labels/label.npz")
        else:
            numpy_file = np.load(path)

        self.y_train_raw = numpy_file['train']
        self.y_test_raw = numpy_file['test']

        print("Labels successfully loaded.")

    def train_dense_model(self, layers: int = 3, units: int = 200):
        input_word_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_type_ids')
        x = self.bert_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)
        x = x[0]
        for _ in range(layers):
            x = Dense(units, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(2, activation='softmax')(x)
        self.bert_dense_model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, input_type_ids],
                                                      outputs=x)

        self.bert_dense_model.compile(loss='binary_crossentropy',
                                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                      metrics=['accuracy'])

        print("Model compiled with summary ----- ")
        print(self.bert_dense_model.summary())

        self.train_history['bert_dense'] = self.bert_dense_model.fit(self.X_train_bert_tokens,
                                                                     self.y_train_raw,
                                                                     epochs=self.EPOCHS)

    def train_lstm_model(self, layers: int = 3, units: int = 64):
        input_word_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_type_ids')
        x = self.bert_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)
        x = x[0]
        for _ in range(layers - 1):
            x = LSTM(units, return_sequences=True)(x)
        x = LSTM(units)(x)
        x = Dense(units, activation='relu')(x)
        x = Dense(2, activation='softmax')(x)
        self.bert_lstm_model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, input_type_ids],
                                                     outputs=x)

        self.bert_lstm_model.compile(loss='binary_crossentropy',
                                     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                     metrics=['accuracy'])

        self.train_history['bert_lstm'] = self.bert_lstm_model.fit(self.X_train_bert_tokens,
                                                                   self.y_train_raw,
                                                                   epochs=self.EPOCHS,
                                                                   )

        print("Model compiles with summary ----- ")
        print(self.bert_lstm_model.summary())

    def train_bilstm_model(self, layers: int = 3, units: int = 64):
        input_word_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_type_ids')
        x = self.bert_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)
        x = x[0]
        for _ in range(layers - 1):
            x = Bidirectional(LSTM(units, return_sequences=True))(x)
        x = Bidirectional(LSTM(units))(x)
        x = Dense(units, activation='relu')(x)
        x = Dense(2, activation='softmax')(x)
        self.bert_bilstm_model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, input_type_ids],
                                                       outputs=x)

        self.bert_bilstm_model.compile(loss='binary_crossentropy',
                                       optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                       metrics=['accuracy'])

        self.train_history['bert_bilstm'] = self.bert_bilstm_model.fit(self.X_train_bert_tokens,
                                                                       self.y_train_raw,
                                                                       epochs=self.EPOCHS,
                                                                       )

        print("Model compiles with summary ----- ")
        print(self.bert_bilstm_model.summary())

    def train_cnn_model(self, units: int = 512):
        input_word_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_type_ids')
        x = self.bert_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)
        x = x[0]
        x = tf.keras.layers.Reshape((self.MAX_SEQ_LENGTH, self.embed_len, 1))(x)
        x = Conv2D(units, kernel_size=(3, self.embed_len), padding='valid',
                   kernel_initializer='normal', activation='relu')(x)
        x = tf.keras.layers.MaxPool2D((2, 1), (2, 1))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(2, activation='softmax')(x)

        self.bert_cnn_model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, input_type_ids],
                                                    outputs=x)

        self.bert_cnn_model.compile(loss='binary_crossentropy',
                                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                    metrics=['accuracy'])

        print("Model compiles with summary ----- ")
        print(self.bert_cnn_model.summary())

        self.train_history['bert_cnn'] = self.bert_cnn_model.fit(self.X_train_bert_tokens,
                                                                 self.y_train_raw,
                                                                 epochs=self.EPOCHS,
                                                                 )

    def train_cnn_bilstm(self):
        input_word_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_type_ids')
        x = self.bert_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)
        x = x[0]

        reshape_layer = tf.keras.layers.Reshape((self.MAX_SEQ_LENGTH, self.embed_len, 1))(x)
        cnn_layer = Conv2D(128, kernel_size=(3, self.embed_len), padding='valid',
                           kernel_initializer='normal', activation='relu')(reshape_layer)
        pool_layer = MaxPool2D((2, 1), strides=(2, 1),
                               padding='valid')(cnn_layer)
        flatten_0 = Flatten()(pool_layer)

        bilstm_layer = Bidirectional(LSTM(128, return_sequences=True))(x)
        bilstm_layer = Bidirectional(LSTM(128, return_sequences=True))(bilstm_layer)
        bilstm_layer = Bidirectional(LSTM(128, return_sequences=True))(bilstm_layer)
        flatten_1 = Flatten()(bilstm_layer)

        merged_layer = tf.keras.layers.Concatenate(axis=1)([flatten_0, flatten_1])
        dense_layer = Dense(512, activation='relu')(merged_layer)
        dropout_layer = tf.keras.layers.Dropout(0.3)(dense_layer)
        output_layer = Dense(2, activation='softmax')(dropout_layer)

        self.cnn_bilstm_model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, input_type_ids],
                                                      outputs=output_layer)

        self.cnn_bilstm_model.compile(loss='binary_crossentropy',
                                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                      metrics=['accuracy'])

        print("Model compiled with summary ----")
        print(self.cnn_bilstm_model.summary())

        self.cnn_bilstm_model.fit(self.X_train_bert_tokens, self.y_train_raw,
                                  epochs=self.EPOCHS)

    def train_multi_cnn(self):
        input_word_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_type_ids')
        x = self.bert_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)
        x = x[0]
        reshape_layer = tf.keras.layers.Reshape((self.MAX_SEQ_LENGTH, self.embed_len, 1))(x)

        cnn_1 = Conv2D(128, kernel_size=(2, self.embed_len), padding='valid',
                       kernel_initializer='normal', activation='relu')(reshape_layer)
        cnn_2 = Conv2D(128, kernel_size=(3, self.embed_len), padding='valid',
                       kernel_initializer='normal', activation='relu')(reshape_layer)
        cnn_3 = Conv2D(128, kernel_size=(4, self.embed_len), padding='valid',
                       kernel_initializer='normal', activation='relu')(reshape_layer)
        cnn_4 = Conv2D(128, kernel_size=(5, self.embed_len), padding='valid',
                       kernel_initializer='normal', activation='relu')(reshape_layer)

        pool_1 = MaxPool2D((2, 1), strides=(2, 1),
                           padding='valid')(cnn_1)
        pool_2 = MaxPool2D((2, 1), strides=(2, 1),
                           padding='valid')(cnn_2)
        pool_3 = MaxPool2D((2, 1), strides=(2, 1),
                           padding='valid')(cnn_3)
        pool_4 = MaxPool2D((2, 1), strides=(2, 1),
                           padding='valid')(cnn_4)

        merged_layer = tf.keras.layers.Concatenate(axis=1)([pool_1, pool_2, pool_3, pool_4])
        flatten_layer = Flatten()(merged_layer)
        dense_layer = Dense(512, activation='relu')(flatten_layer)
        dropout_layer = tf.keras.layers.Dropout(0.3)(dense_layer)
        output_layer = Dense(2, activation='softmax')(dropout_layer)

        self.multi_cnn_model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, input_type_ids],
                                                     outputs=output_layer)
        self.multi_cnn_model.compile(loss='binary_crossentropy',
                                     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                     metrics=['accuracy'])

        print("Model compiled with summary ----")
        print(self.multi_cnn_model.summary())

        self.multi_cnn_model.fit(self.X_train_bert_tokens, self.y_train_raw,
                                 epochs=self.EPOCHS)

    def train_multi_cnn_bilstm(self):
        input_word_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(self.MAX_SEQ_LENGTH,), dtype=tf.int32, name='input_type_ids')
        x = self.bert_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)
        x = x[0]
        reshape_layer = tf.keras.layers.Reshape((self.MAX_SEQ_LENGTH, self.embed_len, 1))(x)

        cnn_1 = Conv2D(128, kernel_size=(2, self.embed_len), padding='valid',
                       kernel_initializer='normal', activation='relu')(reshape_layer)
        cnn_2 = Conv2D(128, kernel_size=(3, self.embed_len), padding='valid',
                       kernel_initializer='normal', activation='relu')(reshape_layer)
        cnn_3 = Conv2D(128, kernel_size=(4, self.embed_len), padding='valid',
                       kernel_initializer='normal', activation='relu')(reshape_layer)
        cnn_4 = Conv2D(128, kernel_size=(5, self.embed_len), padding='valid',
                       kernel_initializer='normal', activation='relu')(reshape_layer)

        pool_1 = MaxPool2D((2, 1), strides=(2, 1),
                           padding='valid')(cnn_1)
        pool_2 = MaxPool2D((2, 1), strides=(2, 1),
                           padding='valid')(cnn_2)
        pool_3 = MaxPool2D((2, 1), strides=(2, 1),
                           padding='valid')(cnn_3)
        pool_4 = MaxPool2D((2, 1), strides=(2, 1),
                           padding='valid')(cnn_4)

        merged_0 = tf.keras.layers.Concatenate(axis=1)([pool_1, pool_2, pool_3, pool_4])
        flatten_0 = Flatten()(merged_0)

        bilstm_layer = Bidirectional(LSTM(128, return_sequences=True))(x)
        bilstm_layer = Bidirectional(LSTM(128, return_sequences=True))(bilstm_layer)
        bilstm_layer = Bidirectional(LSTM(128, return_sequences=True))(bilstm_layer)
        flatten_1 = Flatten()(bilstm_layer)

        merged_layer = tf.keras.layers.Concatenate(axis=1)([flatten_0, flatten_1])
        dense_layer = Dense(512, activation='relu')(merged_layer)
        dropout_layer = tf.keras.layers.Dropout(0.3)(dense_layer)
        output_layer = Dense(2, activation='softmax')(dropout_layer)

        self.multi_cnn_bilstm_model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, input_type_ids],
                                                            outputs=output_layer)
        self.multi_cnn_bilstm_model.compile(loss='binary_crossentropy',
                                            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                            metrics=['accuracy'])

        print("Model compiled with summary ----")
        print(self.multi_cnn_bilstm_model.summary())

        self.multi_cnn_bilstm_model.fit(self.X_train_bert_tokens, self.y_train_raw,
                                        epochs=self.EPOCHS)

    def save_model(self, model, filename: str):
        dir = self.save_path + "/Models-bert/" + filename + ".keras"
        os.makedirs(self.save_path + "/Models-bert/", exist_ok=True)
        try:
            model.save(dir)
            print("Successfully saved model at - ", dir)
        except Exception as e:
            print('There was an error saving model at - ', dir)
            print(e)

    def load_model(self, filepath: str):
        try:
            model = tf.keras.models.load_model(filepath)
            print("Loaded model {} from path {}".format(model, filepath))
            print("Summary - {}".format(model.summary()))

            return model
        except Exception as e:
            print(e)

    def evaluate_model(self, model):
        try:
            y_test = self.y_test_raw
            y_pred = np.round(model.predict(self.X_test_bert_tokens))

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test[:, 0], y_pred[:, 0])
            recall = recall_score(y_test[:, 0], y_pred[:, 0])
            f1 = f1_score(y_test[:, 0], y_pred[:, 0])
            roc_auc = roc_auc_score(y_test[:, 0], y_pred[:, 0])

            print("Accuracy - ", accuracy)
            print("Precision - ", precision)
            print("Recall - ", recall)
            print("F1 - ", f1)
            print("Roc-Auc - ", roc_auc)

            return accuracy, precision, recall, f1, roc_auc
        except Exception as e:
            print(e)

    def get_bert_tokens(self, texts):
        texts = [self.seq_preprocess(seq) for seq in texts]
        ct = len(texts)
        input_ids = np.ones((ct, self.MAX_SEQ_LENGTH), dtype='int32')
        attention_mask = np.zeros((ct, self.MAX_SEQ_LENGTH), dtype='int32')
        token_type_ids = np.zeros((ct, self.MAX_SEQ_LENGTH), dtype='int32')

        for k, text in tqdm(enumerate(texts), total=len(texts)):
            # Tokenize
            tok_text = self.bert_tokenizer.tokenize(text)

            # Truncate and convert tokens to numerical IDs
            enc_text = self.bert_tokenizer.convert_tokens_to_ids(tok_text[:(self.MAX_SEQ_LENGTH - 2)])

            input_length = len(enc_text) + 2
            input_length = input_length if input_length < self.MAX_SEQ_LENGTH else self.MAX_SEQ_LENGTH

            # Add tokens [CLS] and [SEP] at the beginning and the end
            input_ids[k, :input_length] = np.asarray([0] + enc_text + [2], dtype='int32')

            # Set to 1s in the attention input
            attention_mask[k, :input_length] = 1

        return {
            'input_word_ids': tf.constant(input_ids),
            'input_mask': tf.constant(attention_mask),
            'input_type_ids': tf.constant(token_type_ids)
        }

    def seq_preprocess(self, sequence: str):
        processed_words = simple_preprocess(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(#[A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", sequence))
        processed_words = [word.lower() for word in processed_words if word.lower() not in self.stop_words]

        return ' '.join(processed_words)