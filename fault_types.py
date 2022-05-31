import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
from sklearn.metrics import adjusted_mutual_info_score 
from sklearn.metrics import completeness_score

def fault_type_sensor_classifier(st, X_train, y_train, signal_length, signal_channel, epochs,  batch_size, verbose):
    st_mlp = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, 
            activation=tf.keras.activations.gelu),
        tf.keras.layers.Dense(len(y_train[0])),
    ]
    )
    inputs = tf.keras.layers.Input((signal_length, signal_channel))
    x = st(inputs)
    x = x[:, 0]
    outputs = st_mlp(x)
    st_classifier = tf.keras.models.Model(inputs, outputs)
    st_classifier.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
          optimizer=tf.keras.optimizers.Adam(),
          metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )
    st_classifier.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return st_classifier

def fault_type_AE_classifier(model, layer_name, X_train, y_train, epochs,  batch_size, verbose):
    x = model.get_layer(layer_name).output
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    out =  tf.keras.layers.Dense(len(y_train[0]), activation='linear')(x)
    extracted_model = Model(inputs=model.input, outputs=out)
    extracted_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()])
    extracted_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    #print(extracted_model.summary())
    x = extracted_model.get_layer('global_max_pooling1d_1').output
    emb_model = tf.keras.models.Model(extracted_model.input, x)
    return emb_model

def extract_embeddings(model, layer_name):
    st = model.get_layer(layer_name)
    x = st.output
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    embedding_model = tf.keras.models.Model(st.input, x)
    return embedding_model


def extract_embeddings_AE(model, layer_name):
    x = model.get_layer(layer_name).output
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    embedding_model = tf.keras.models.Model(model.input, x)
    return embedding_model

def clutering_k_mean(train_rep, test_rep, number_cluster, y_test):
    kmeans = KMeans(n_clusters=number_cluster, max_iter=5000,  n_init=50, random_state=0).fit(train_rep)
    test_result = kmeans.predict(test_rep)
    ami =  adjusted_mutual_info_score(y_test, test_result)
    h = homogeneity_score(y_test, test_result)
    c = completeness_score(y_test, test_result)
    return ami, h, c


def fault_type_superviser_classifier(X_train, y_train, epochs, batch_size, verbose):
    model = tf.keras.Sequential(
    [
        layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tfa.layers.InstanceNormalization(axis=2, 
            epsilon=1e-6,
            center=False, 
            scale=False, 
            beta_initializer="glorot_uniform",
            gamma_initializer="glorot_uniform"),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.1),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.1),
        layers.Conv1D(
            filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.1),
        layers.Conv1D(
            filters=128, kernel_size=7, padding="same", strides=2, activation="relu", name = 'encoder_lastlayer'
        ),
        layers.GlobalMaxPooling1D(),
        layers.Dense(len(y_train[0]), activation='linear')
    ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model