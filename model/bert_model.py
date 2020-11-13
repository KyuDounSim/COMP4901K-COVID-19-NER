"""model definition"""

# TODO: Clean up unused modules

import os

import matplotlib.pyplot as plt
import numpy as np

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

import tensorflow as tf

import tensorflow_hub as hub


def build_Bert_token_classifier(model_name,
                                output_size,
                                output_layer,
                                output_activation,
                                time_distrib=False,
                                bidirectional=True,
                                seq_length=None,
                                dropout_rate=0.1):
    """
    Code sources:
    1. https://github.com/tensorflow/models/blob/master/official/nlp/bert/bert_models.py
    2. https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3
    3. https://www.tensorflow.org/official_models/fine_tuning_bert
    4. https://github.com/bhuvanakundumani/BERT-NER-TF2/
    5. https://github.com/dmis-lab/biobert

    Builds and returns a BERT model with the specified output layer on top.
    The options are:
    # 1. Dense / Time Distributed dense
    # 2. LSTM / Bi-LSTM / CRF-LSTM / CRF-Bi-LSTM
    # 3. Seq2Seq, etc.

    The model is a keras model.
    """
    x = dict(input_word_ids=tf.keras.layers.Input(shape=(seq_length, ),
                                                  dtype=tf.int32),
             input_mask=tf.keras.layers.Input(shape=(seq_length, ),
                                              dtype=tf.int32),
             input_type_ids=tf.keras.layers.Input(shape=(seq_length, ),
                                                  dtype=tf.int32))

    # TODO: move toward loading from saved checkpoint, for more flexibility.
    # Also, we might want to use bio-bert: https://github.com/dmis-lab/biobert
    hub_encoder = hub.KerasLayer(f'https://tfhub.dev/tensorflow/{model_name}',
                                 trainable=True,
                                 name='BERT')
    h = hub_encoder(x, training=True)
    h = h['sequence_output']  # [N, T, 768]

    # NOTE: Moving toward BioBert, which do not mask the invalid tokens
    # # valid_ids
    # v = tf.keras.layers.Input(shape=(seq_length, ), dtype=tf.int32)
    # mask = tf.cast(v, dtype=tf.bool)

    # def _components(args):
    #     values, bool_mask = args
    #     ragged = tf.ragged.boolean_mask(values, bool_mask)
    #     return ragged

    # masked = tf.keras.layers.Lambda(_components)([h, mask])
    # zero_at_end = masked.to_tensor(default_value=0)  # [N, num_valid_ids, 768]

    # padded = tf.pad(
    #   zero_at_end, [[0,0], [0, seq_length - tf.shape(zero_at_end)[1]], [0,0]])
    # padded = tf.reshape(padded, [-1, seq_length, 768])  # hotfix for None dim.

    d_h = tf.keras.layers.Dropout(rate=dropout_rate)(h)

    if output_layer == 'dense':
        classifier = tf.keras.layers.Dense(
            output_size,
            activation=output_activation,
            kernel_initializer=tf.keras.initializers.he_normal(seed=0),
            bias_initializer='zeros')
        if time_distrib:
            classifier = tf.keras.layers.TimeDistributed(classifier)

    elif output_layer == 'lstm':
        raise NotImplementedError

    elif output_layer == 'gru':
        raise NotImplementedError

    elif output_layer == 'seq2seq':
        raise NotImplementedError

    else:
        raise NotImplementedError

    y = classifier(d_h)

    model = tf.keras.models.Model(x, y)

    return model


def train(model,
          train_data,
          train_target,
          val_data,
          val_target,
          train_data_size,
          batch_size=32,
          epochs=5,
          lr=2e-5):

    # TODO: Need checkpointing call back, save / load code.
    steps_per_epoch = int(train_data_size / batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

    # creates an optimizer with learning rate schedule
    optimizer = nlp.optimization.create_optimizer(
        lr, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy('accuracy',
                                                   dtype=tf.float32)
    ]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = model.fit(train_data,
                        train_target,
                        validation_data=(val_data, val_target),
                        batch_size=batch_size,
                        epochs=epochs)

    return model, history


#TODO: predict code
