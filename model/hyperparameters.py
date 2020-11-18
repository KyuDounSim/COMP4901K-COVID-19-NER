"""Declare all hyperparameters in this file."""


class HP:
    """Hyper parameters"""
    # TODO: Add more parameters to control
    # bert model name.
    # cased = use case distinction. It is known to be better for NER task.
    # L = number of tranformer_encoder layer
    # H = hidden dimension for each encoder
    # A = number of attention heads for each self attention
    # /3 = version number. Most tutorials are using V2, but why not use V3?
    model_name = 'cased_L-12_H-768_A-12'  # or biobert
    hub_model_name = f'bert_en_{model_name}/3'
    bert_dir = f'pre_bert/tf2_{model_name}'
    data_dir = 'data'
    # model related
    # should be one of dense, lstm, crf, softmax, (seq2seq, and others based on
    # our trials)
    output_layer = 'dense'  # [dense, lstm, gru, seq2seq]
    output_activation = 'softmax'  # [softmax, crf]
    time_distrib = False  # used for dense
    bidirectional = True  # used for lstm

    # training related
    epochs = 3
    batch_size = 32
    eval_batch_size = 32
    dropout_rate = 0.1
    learning_rate = 2e-5

    # dataset related
    train_data_size = 8192  # dummy number: to be deleted later with real input
    max_seq_len = 128  # to be decided later with the dataset
