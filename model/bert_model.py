"""model definition"""

# TODO: Clean up unused modules

import os
import json

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

import pandas as pd
import tensorflow as tf

# import tensorflow_hub as hub


def build_Bert_token_classifier(model_dir,
                                output_size,
                                output_layer,
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
    x = dict(input_word_ids=tf.keras.layers.Input(shape=(seq_length, ), dtype=tf.int32),
             input_mask=tf.keras.layers.Input(shape=(seq_length, ), dtype=tf.int32),
             input_type_ids=tf.keras.layers.Input(shape=(seq_length, ), dtype=tf.int32))

    bert_config_path = os.path.join(model_dir, 'bert_config.json')
    with open(bert_config_path, 'r') as bert_config_file:
        config_dict = json.loads(bert_config_file.read())

    bert_config = bert.configs.BertConfig.from_dict(config_dict)
    bert_encoder = bert.bert_models.get_transformer_encoder(bert_config,
                                                            sequence_length=seq_length)
    h = bert_encoder(x, training=False)[0]  # [N, T, 768]

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
            activation=None,  # the activation is None, since the loss is from_logits
            kernel_initializer=tf.keras.initializers.he_normal(seed=0),
            bias_initializer='zeros')
        if time_distrib:
            classifier = tf.keras.layers.TimeDistributed(classifier)
        y = classifier(d_h)

    elif output_layer == 'lstm':
        classifier = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            		output_size,
	            activation='tanh',
            		recurrent_activation = 'sigmoid',
            		kernel_initializer=tf.keras.initializers.he_normal(seed=0),
            		bias_initializer='zeros', return_sequences= True), 
            		merge_mode = 'sum')
        lstm_output = classifier(d_h)
        y = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size))(lstm_output)

    elif output_layer == 'gru':
        raise NotImplementedError

    elif output_layer == 'seq2seq':
        raise NotImplementedError

    else:
        raise NotImplementedError

    model = tf.keras.models.Model(x, y)

    return model, bert_encoder


def train(model,
          bert_encoder,
          model_dir,
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

    # Load from pre-trained checkpoint
    checkpoint = tf.train.Checkpoint(model=bert_encoder)
    checkpoint.restore(os.path.join(model_dir, 'bert_model.ckpt')).assert_consumed()

    # creates an optimizer with learning rate schedule
    optimizer = nlp.optimization.create_optimizer(lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=warmup_steps)

    # NOTE: If using SparseCategoricalCrossentropy, the label should not be
    # one-hot vector. Not of shape [N, T, C], but [N, T, 1], where each class are
    # integer values. This is used in the original code, so we should experiment
    # that also.
    metrics = [tf.keras.metrics.CategoricalAccuracy('accuracy', dtype=tf.float32)]
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = model.fit(train_data,
                        train_target,
                        validation_data=(val_data, val_target),
                        batch_size=batch_size,
                        epochs=epochs)

    return model, history


def evaluate(model, inp, target, mask, tag2id, targ_seq_len=128):
    """evaluation code.

    args:
        model: trained model. Keras model.
        inp: input to evaluate on (preprocessed). Numpy Array.
        target: target to evaluate on (non-preprocessed raw target). Python List.
        mask: label mask: applied on predicted output. Numpy array.
        tag2id: change from string tag 2 id. A python Dictionary.
        targ_seq_len: original sequence length of original dataset. 128
    """
    output = model.predict(inp, batch_size=64)  # [N, max_seq_len, 68]
    softmax_o = tf.keras.layers.Softmax()(output)  # [N, max_seq_len]

    raw_pred = tf.argmax(softmax_o, axis=-1)  # [N, max_seq_len]

    pred = tf.ragged.boolean_mask(raw_pred, mask).to_tensor(default_value=0).numpy()

    if pred.shape[1] < targ_seq_len:
        # [N, targ_seq_len]
        pred = np.pad(pred, [[0, 0], [0, targ_seq_len - pred.shape[1]]])

    # target tag -> integer
    targ = np.array([[tag2id[tag] for tag in ex] for ex in target])  # [N, targ_seq_len]
    assert pred.shape == targ.shape

    p_f = pred.flatten()
    t_f = targ.flatten()

    non_inv_idx = np.where(t_f != tag2id["_t_pad_"])[0]
    return sum(p_f[non_inv_idx] == t_f[non_inv_idx]) / len(non_inv_idx)


def predict(model, inp, mask, id2tag, fname='test_preds.csv', targ_seq_len=128):
    """prediction code.

    args:
        model: trained model. Keras model.
        inp: input to evaluate on (preprocessed). Numpy Array.
        target: target to evaluate on (non-preprocessed raw target). Python List.
        mask: label mask: applied on predicted output. Numpy array.
        id2tag: A Python dictionary that maps the pred id to string tag.
        targ_seq_len: original sequence length of original dataset. 128
    """
    output = model.predict(inp, batch_size=64)  # [N, max_seq_len, 68]
    softmax_o = tf.keras.layers.Softmax()(output)  # [N, max_seq_len]

    raw_pred = tf.argmax(softmax_o, axis=-1)  # [N, max_seq_len]
    pred = tf.ragged.boolean_mask(raw_pred, mask).to_tensor(default_value=0).numpy()

    if pred.shape[1] < targ_seq_len:
        # [N, targ_seq_len]
        pred = np.pad(pred, [[0, 0], [0, targ_seq_len - pred.shape[1]]])

    # pred
    pred_tag = [[id2tag[tag] for tag in ex] for ex in pred.tolist()]  # [N, targ_seq_len]
    df = pd.DataFrame({
        'id': np.arange(len(pred_tag)),
        'labels': [json.dumps(preds) for preds in pred_tag]
    })
    df.to_csv(fname, index=False)
    return pred_tag
