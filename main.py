#!/usr/bin/env python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from transformations_1c import get_transformations
from network import get_ssl_network
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

from fault_types import extract_embeddings, clutering_k_mean, fault_type_sensor_classifier

def data_generator(X, batch_size, transformations):
    n = len(X)
    ix = np.arange(n)
    while True:
        np.random.shuffle(ix)
        for i in range(n // batch_size):
            _ix = ix[i * batch_size:(i + 1) * batch_size]
            _X = X[_ix]
            _X_aug = []
            for _xin in _X:
                aug_fn = np.random.choice(transformations)
                _x_aug = aug_fn(_xin)
                _X_aug.append(_x_aug)

            _X_aug = np.stack(_X_aug, axis = 0)
            _X, _X_aug = shuffle(_X, _X_aug)
            _X = _X.astype("float32")
            _X_aug = _X_aug.astype("float32")
            yield _X, _X_aug 


def pretraining_sensor_transformation(x_train, signal_length, segment_size, signal_channel, epochs,  batch_size, verbose):
    transformations = get_transformations()
    data_gen = data_generator(x_train, batch_size, transformations)
    model = get_ssl_network(signal_length=signal_length, 
        segment_size=segment_size, 
        signal_channels=signal_channel, 
        code_size=64, 
        l2_rate=1e-4)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4))
    model.fit(data_gen, epochs=epochs, 
        steps_per_epoch=x_train.shape[0] // batch_size, verbose=verbose) 
    return model


def sensor_classifier(st, X_train, y_train, signal_length, signal_channel, epochs,  batch_size, verbose):
    st_mlp = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, 
            activation=tf.keras.activations.gelu),
        tf.keras.layers.Dense(len(np.unique(y_train))),
    ]
    )
    inputs = tf.keras.layers.Input((signal_length, signal_channel))
    x = st(inputs)
    x = x[:, 0]
    outputs = st_mlp(x)
    st_classifier = tf.keras.models.Model(inputs, outputs)
    st_classifier.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          optimizer=tf.keras.optimizers.Adam(),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
    st_classifier.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return st_classifier

def sensor_transformer_complete(data_path, model_path, result_path,
        X_train, y_train, X_test, y_test, signal_length, 
        segment_size, signal_channel, epochs, batch_size, verbose):
    model = tf.keras.models.load_model(model_path+'/data_efficiency/')
    em = model.get_layer('embedding_model')
    st = em.get_layer('sensor_transformer')
    model = sensor_classifier(st, X_train, y_train, signal_length, signal_channel, epochs,  batch_size, verbose)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis = 1)
    acc = accuracy_score(y_test, y_pred)
    f = f1_score(y_test, y_pred,  average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)
    pd.DataFrame({'Acc': [acc], 'fscores': [f], 'kappa': [kappa]}).to_csv(result_path+'/pretrained_ST_complete_ds.csv')



def data_efficiency_sensor_transformation(data_path, model_path, result_path,
                        X_train, X_test, y_test, signal_length, segment_size, signal_channel, epochs, batch_size, verbose):
    
    model = pretraining_sensor_transformation(model_path, X_train, signal_length, segment_size, signal_channel, epochs,  batch_size, verbose)
    #model.save(model_path+'/data_efficiency/')
    #model = tf.keras.models.load_model(model_path+'/data_efficiency/')
    em = model.get_layer('embedding_model')
    st = em.get_layer('sensor_transformer')

    print('**************Model Pretrained************')

    accuracies = []
    f_scores = []
    kappas = []
    number_samples = [5, 10, 20, 50, 100]
    for sample in number_samples:
        print('***********Current sample**********', sample)
        X_train = np.load(data_path+'_processed/number_samples/'+str(sample)+'_X_train.npy')
        y_train = np.load(data_path+'_processed/number_samples/'+str(sample)+'_y_train.npy')
        model = sensor_classifier(st, X_train, y_train, signal_length, signal_channel, epochs,  batch_size, verbose)
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis = 1)
        acc = accuracy_score(y_test, y_pred)
        f = f1_score(y_test, y_pred,  average='weighted')
        kappa = cohen_kappa_score(y_test, y_pred)
        #l, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
        accuracies.append(acc)
        f_scores.append(f)
        kappas.append(kappa)
    pd.DataFrame({'number_samples': number_samples, 'Acc': accuracies, 
        	                                        'fscores': f_scores, 'kappa': kappas}).to_csv(result_path+'/data_efficiency/pretrained_ST_cutout.csv')



def fault_types_sensor_transformation(data_path, model_path, result_path, fault_list,
                        X_test, y_test, signal_length, segment_size, signal_channel, epochs, batch_size, verbose):
    amis, hs, cs, p_amis, p_hs, p_cs = [], [], [], [], [], []
    for fault in fault_list:
        X_train = np.load(data_path+'_processed/fault_types/'+str(fault)+'/X_train.npy')
        print(X_train.shape, fault)
        model = pretraining_sensor_transformation(model_path, X_train, signal_length, segment_size, signal_channel, epochs,  batch_size, verbose)
        model.save(model_path+'/fault_types/Pretrained/'+fault+'/')

    for fault in fault_list:
        X_train = np.load(data_path+'_processed/fault_types/'+str(fault)+'/X_train.npy')
        y_train = np.load(data_path+'_processed/fault_types/'+str(fault)+'/y_train.npy')

        X_test = np.load(data_path+'_processed/train_test/X_test.npy')
        print(X_test.shape)
        y_test = np.load(data_path+'_processed/train_test/y_test.npy')
        
        model = tf.keras.models.load_model(model_path+'/fault_types/Pretrained/'+str(fault)+'/')
        em = model.get_layer('embedding_model')
        em_model = extract_embeddings(em, em.layers[1].name)
        ami, h, c = clutering_k_mean(em_model.predict(X_train), 
                                                    em_model.predict(X_test), len(np.unique(y_train)), y_test)
        amis.append(ami)
        hs.append(h)
        cs.append(c)

        yy_train = to_categorical(y_train)
        em = model.get_layer('embedding_model')
        print(em.layers[1].name)
        st = em.get_layer(em.layers[1].name)
        yy_train = to_categorical(y_train)
        model_cls = fault_type_sensor_classifier(st, X_train, yy_train, signal_length, signal_channel, epochs,  batch_size, verbose)
        model_cls.save(model_path+'/fault_types/Classifier/'+fault+'/')
        em_model_cls = extract_embeddings(model_cls, model_cls.layers[1].name)
        amip, hp, cp = clutering_k_mean(em_model_cls.predict(X_train), 
                                                    em_model_cls.predict(X_test), len(np.unique(y_train)), y_test)

        p_amis.append(amip)
        p_hs.append(hp)
        p_cs.append(cp)
    pd.DataFrame({'fault_type': fault_list, 'ami': amis, 'h': hs, 'c': cs, 
                                            'p_ami': p_amis, 'p_hs': p_hs, 'p_cs': p_cs}).to_csv(result_path+'/fault_types/pretrained_ST_fault_types.csv')
    

if __name__ == "__main__":

    dataset_name = 'KAT' #48kDE_CWRU' #
    data_path = '../Data/'+dataset_name
    model_path = '../Model/'+dataset_name
    result_path = '../Result/'+dataset_name


    epochs = 100
    batch_size = 24
    verbose = 2

    X_train = np.load(data_path+'_processed/train_test/X_train.npy')
    y_train = np.load(data_path+'_processed/train_test/y_train.npy')
    X_test = np.load(data_path+'_processed/train_test/X_test.npy')
    y_test = np.load(data_path+'_processed/train_test/y_test.npy')

    signal_length = X_train.shape[1]
    signal_channel = X_train.shape[2]
    if dataset_name == 'KAT':
        segment_size = 150
    else:
        segment_size = 64
    
 
    data_efficiency_sensor_transformation(data_path, model_path, result_path,
                        X_train, X_test, y_test, signal_length, segment_size, signal_channel, epochs, batch_size, verbose)

    fault_list = ['B', 'IR', 'OR']
    fault_types_sensor_transformation(data_path, model_path, result_path, fault_list, X_test, y_test, 
                                                    signal_length, segment_size, signal_channel, epochs, batch_size, verbose)

    sensor_transformer_complete(data_path, model_path, result_path,
                        X_train, y_train, X_test, y_test, signal_length, segment_size, signal_channel, epochs, batch_size, verbose)

    









