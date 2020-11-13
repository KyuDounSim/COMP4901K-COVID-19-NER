"""A module dedicated to preprocessing the dataset.

BERT requires specific data input types: need to conform to it.
Also, we need to use the same bert_tokenizer used for the bert model we are
using.
"""

from itertools import chain
import os
import pickle as pkl

import numpy as np
from official import nlp
from official.nlp import bert
import official.nlp.bert.tokenization
from tensorflow.keras.utils import to_categorical

from model.hyperparameters import HP




def load_data():
    train_dict = pkl.load(open('data/train.pkl', 'rb'))
    val_dict = pkl.load(open('data/val.pkl', 'rb'))
    test_dict = pkl.load(open('data/test.pkl', 'rb'))
    print('keys in train_dict:', train_dict.keys())
    print('keys in val_dict:', val_dict.keys())
    print('keys in test_dict:', test_dict.keys())
    return train_dict, val_dict, test_dict


def convert_examples_to_features(word_seq,
                                 tag_seq,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 print_ex=False):
    """Loads a data file into a list of `InputBatch`s.

    This function is forked from:
    https://github.com/bhuvanakundumani/BERT-NER-TF2/blob/master/run_ner.py
    """

    # TODO: findout what happens to Unkonwn tokens (since the BERT tokenizer
    # would not hvae seen much of bio related entities, they might just all be
    # [UNK] tokens. we need to circumvent this. May wanna use BioBERT for this reason.)
    # TODO: deal with the _w_pad_ tokens and _t_pad_ tags. Might wanna just
    # delete all of them. It is okay to delete them since they are filtered in
    # evaluate code anyway.
    label_map = {label: i for i, label in enumerate(label_list, 1)}

    inp = {
        'input_ids': [],
        'input_mask': [],
        'segment_ids': [],
        # 'valid_ids': [],
    }
    target = {
        'label_ids': [],
        # 'label_mask': [],
    }
    for (ex_index, example) in enumerate(word_seq):
        textlist = example
        labellist = tag_seq[ex_index]
        tokens = []
        labels = []
        # valid_ids = []
        # label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    # valid_ids.append(1)
                    # label_mask.append(True)
                else:
                    # valid_ids.append(0)
                    labels.append('[INV]')
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid_ids = valid_ids[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append('[CLS]')
        segment_ids.append(0)
        # valid_ids.insert(0, 1)
        # label_mask.insert(0, True)
        label_ids.append(label_map['[CLS]'])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append('[SEP]')
        segment_ids.append(0)
        # valid_ids.append(1)
        # label_mask.append(True)
        label_ids.append(label_map['[SEP]'])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        # label_mask = [True] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            # valid_ids.append(1)
            # label_mask.append(False)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            # label_mask.append(False)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        # assert len(valid_ids) == max_seq_length
        # assert len(label_mask) == max_seq_length

        if ex_index < 5 and print_ex:
            print('*** Example ***')
            print('tokens: %s' % ' '.join([str(x) for x in tokens]))
            print('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
            print('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
            print('segment_ids: %s' % ' '.join([str(x) for x in segment_ids]))

        inp['input_ids'].append(input_ids)
        inp['input_mask'].append(input_mask)
        inp['segment_ids'].append(segment_ids)
        # inp['valid_ids'].append(valid_ids)

        target['label_ids'].append(label_ids)
        # target['label_mask'].append(label_mask)

    return inp, target


def get_training_data():
    train_dict, val_dict, test_dict = load_data()

    del test_dict  # unused

    tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(
        HP.bert_dir, 'vocab.txt'),
                                                do_lower_case=False)

    tags = set(chain(*train_dict['tag_seq']))
    tags.add('[CLS]')
    tags.add('[SEP]')
    tags.add('[INV]')  # invalid token introduced from wordpiece tokenizer

    train_inp, train_target = convert_examples_to_features(
        train_dict['word_seq'], train_dict['tag_seq'], tags, HP.max_seq_len,
        tokenizer)

    val_inp, val_target = convert_examples_to_features(val_dict['word_seq'],
                                                       val_dict['tag_seq'],
                                                       tags, HP.max_seq_len,
                                                       tokenizer)

    train_inp = {k: np.array(v) for k, v in train_inp.items()}
    train_target = {
        k: to_categorical(np.array(v), num_classes=len(tags))
        for k, v in train_target.items()
    }
    val_inp = {k: np.array(v) for k, v in val_inp.items()}
    val_target = {
        k: to_categorical(np.array(v), num_classes=len(tags))
        for k, v in val_target.items()
    }

    return train_inp, train_target, val_inp, val_target
