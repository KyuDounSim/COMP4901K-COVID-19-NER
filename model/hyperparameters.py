"""Declare all hyperparameters in this file."""


class HP:
    """Hyper parameters"""
    # TODO: Add more parameters to control
    # bert model name.
    # cased = use case distinction. It is known to be better for NER task.
    # L = number of tranformer_encoder layer
    # H = hidden dimension for each encoder
    # A = number of attention heads for each self attention
    model_name = 'biobert_large'  # ['cased_L-12_H-768_A-12', biobert, biobert_large, bert_large_cased]
    bert_dir = f'pre_bert/tf2_{model_name}'
    data_dir = 'data'
    # model related
    output_layer = 'dense'  # [dense, lstm, gru]
    time_distrib = False  # used for trying time_distributed dense
    bidirectional = True  # used for lstm

    # training related
    epochs = 5
    batch_size = 8
    eval_batch_size = 64
    dropout_rate = 0.1
    learning_rate = 1e-5

    # dataset related
    max_seq_len = 256  # to be decided later with the dataset
    targ_seq_len = 128  # original max seq length of dataset: they are all paddedto 128
