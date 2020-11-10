"""model definition"""

# TODO: Clean up unused modules

import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_hub as hub

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks


def build_Bert_token_classifier(model_name, option, time_distrib=False,
                                bidirectional=True, seq_length=None,
                                output_size=65, dropout_rate=0.1):
    """
    Code sources:
    1. https://github.com/tensorflow/models/blob/master/official/nlp/bert/bert_models.py
    2. https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3
    3. https://www.tensorflow.org/official_models/fine_tuning_bert

    """
    x = dict(
        input_word_ids=tf.keras.layers.Input(shape=(seq_length,),
                                             dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(seq_length,),
                                         dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(seq_length,),
                                             dtype=tf.int32))

    # NOTE: rn, using a hub model, since it is easier. However, this abstracts
    # the inner layers of BERT, so hard to visualize and modify. Later, if we
    # want to modify or take a look at the structure, might have to move toward
    # a non-hub model, which requires downloading from a checkpoint.

    # Also, we might want to use bio-bert: https://github.com/dmis-lab/biobert

    hub_encoder = hub.KerasLayer(f'https://tfhub.dev/tensorflow/{model_name}',
                                 trainable=True, name='BERT')

    h = hub_encoder(x, training=True)

    d_h = tf.keras.layers.Dropout(rate=dropout_rate)(h['sequence_output'])

    # options:
    # 1. Dense / Time Distributed dense
    # 2. LSTM / Bi-LSTM / CRF-LSTM / CRF-Bi-LSTM
    # 3. CRF layer / Softmax Layer
    # 4. Softmax Layer
    # 5. etc. Seq2Seq,

    # These two add a classification head on top of BERT model
    if option == 'dense':
        classifier = tf.keras.layers.Dense(output_size,
                                  activation='softmax',
                                  kernel_initializer=tf.keras.initializers.he_normal(seed=0),
                                  bias_initializer='zeros')
        if time_distrib:
            classifier = tf.keras.layers.TimeDistributed(classifier)

    elif option == 'lstm':
        raise NotImplementedError

    # these 2 are direct finetuning
    elif option == 'crf':
        raise NotImplementedError

    elif option == 'softmax':
        raise NotImplementedError

    else:
        raise NotImplementedError

    y = classifier(d_h)

    model = tf.keras.models.Model(x, y)

    return model


def train(model, train_data, train_target, val_data, val_target,
          train_data_size, batch_size, epochs, lr=2e-5):

    # TODO: Need checkpointing call back, save / load code.
    steps_per_epoch = int(train_data_size / batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

    # creates an optimizer with learning rate schedule
    optimizer = nlp.optimization.create_optimizer(
        lr, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy',
                                                          dtype=tf.float32)]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    history = model.fit(
        train_data, train_target,
        validation_data=(val_data, val_target),
        batch_size=batch_size,
        epochs=epochs)

    return model, history

#TODO: evaluation code
