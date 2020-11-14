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


class DataLoader():
    def __init__(self, data_dir, bert_dir, max_seq_len):
        self.data_dir = data_dir
        self.bert_dir = bert_dir
        self.max_seq_len = max_seq_len

        self.train_dict = {}
        self.val_dict = {}
        self.test_dict = {}

        self.tag2id = {}
        self.id2tag = {}

    def load_data(self):
        train_dict = pkl.load(open(os.path.join(self.data_dir, 'train.pkl'), 'rb'))
        val_dict = pkl.load(open(os.path.join(self.data_dir, 'val.pkl'), 'rb'))
        test_dict = pkl.load(open(os.path.join(self.data_dir, 'test.pkl'), 'rb'))
        print('keys in train_dict:', train_dict.keys())
        print('keys in val_dict:', val_dict.keys())
        print('keys in test_dict:', test_dict.keys())
        self.train_dict = train_dict
        self.val_dict = val_dict
        self.test_dict = test_dict


    def set_tags(self):
        tags = set(chain(*self.train_dict['tag_seq']))
        self.tag2id['_t_pad_'] = 0
        self.tag2id['[INV]'] = 1
        self.tag2id['[CLS]'] = 2
        self.tag2id['[SEP]'] = 3
        for tag in tags:
            if tag not in self.tag2id:
                self.tag2id[tag] = len(self.tag2id)

        self.id2tag = {i: tag for tag, i in self.tag2id.items()}

        print(self.tag2id)


    def convert_examples_to_features(self,
                                    word_seq,
                                    tag_seq,
                                    max_seq_length,
                                    tokenizer,
                                    print_ex=0):
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

        inp = {
            'input_ids': [],
            'input_mask': [],
            'segment_ids': [],
            # 'valid_ids': [],
        }
        target = {
            'label_ids': [],
            'label_mask': [],
        }
        for (ex_index, example) in enumerate(word_seq):
            textlist = example
            labellist = tag_seq[ex_index]
            tokens = []
            labels = []
            # valid_ids = []
            label_mask = []
            for i, word in enumerate(textlist):
                label_1 = labellist[i]
                if word == '_unk_':
                    tokens.append('[UNK]')
                    labels.append(label_1)
                    # valid_ids.append(1)
                    label_mask.append(True)
                    continue
                elif word == '_w_pad_':
                    tokens.append('[PAD]')
                    labels.append(label_1)
                    # valid_ids.append(1)
                    label_mask.append(True)
                    continue
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        # valid_ids.append(1)
                        label_mask.append(True)
                    else:
                        # valid_ids.append(0)
                        labels.append('[INV]')
                        label_mask.append(False)
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                # valid_ids = valid_ids[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]
            ntokens = []
            segment_ids = []
            label_ids = []
            ntokens.append('[CLS]')
            segment_ids.append(0)
            # valid_ids.insert(0, 1)
            label_mask.insert(0, True)
            label_ids.append(self.tag2id['[CLS]'])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    label_ids.append(self.tag2id[labels[i]])
            ntokens.append('[SEP]')
            segment_ids.append(0)
            # valid_ids.append(1)
            label_mask.append(True)
            label_ids.append(self.tag2id['[SEP]'])
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            # label_mask = [True] * len(label_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                # valid_ids.append(1)
                label_mask.append(False)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            # assert len(valid_ids) == max_seq_length
            assert len(label_mask) == max_seq_length

            if ex_index < print_ex:
                print('*** Example ***')
                print('tokens: %s' % ' '.join([str(x) for x in ntokens]))
                print('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
                print('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
                print('segment_ids: %s' % ' '.join([str(x) for x in segment_ids]))
                print('label_ids: %s' % ' '.join([str(x) for x in label_ids]))

            inp['input_ids'].append(input_ids)
            inp['input_mask'].append(input_mask)
            inp['segment_ids'].append(segment_ids)
            # inp['valid_ids'].append(valid_ids)

            target['label_ids'].append(label_ids)
            target['label_mask'].append(label_mask)

        inp = {k: np.array(v) for k, v in inp.items()}
        target = {
            k: to_categorical(np.array(v), num_classes=len(self.tag2id))
            for k, v in target.items()
        }

        return inp, target


    def get_train_data(self, print_ex=0):
        self.load_data()
        self.set_tags()

        tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(
            self.bert_dir, 'vocab.txt'),
                                                    do_lower_case=False)

        train_inp, train_target = self.convert_examples_to_features(
            self.train_dict['word_seq'], self.train_dict['tag_seq'], self.max_seq_len,
            tokenizer, print_ex=print_ex)

        val_inp, val_target = self.convert_examples_to_features(self.val_dict['word_seq'],
                                                        self.val_dict['tag_seq'],
                                                        self.max_seq_len,
                                                        tokenizer, print_ex=print_ex)

        return train_inp, train_target, val_inp, val_target
