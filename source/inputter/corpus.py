# m

import os
import torch
import json
from tqdm import tqdm
from source.inputter.field import tokenize
from source.inputter.field import TextField
from source.inputter.batcher import DialogBatcher


class KnowledgeCorpus(object):
    """
    KnowledgeCorpus
    """
    def __init__(self,
                 data_dir,
                 min_freq=0,
                 max_vocab_size=None,
                 min_len=0,
                 max_len=100,
                 min_kb_len=0,
                 max_kb_len=1000,
                 embed_file=None,
                 share_vocab=False,
                 vocab_type=None):

        self.data_dir = data_dir
        self.prepared_data_file = f"{data_dir}/data.{vocab_type}.pt"
        self.prepared_vocab_file = f"{data_dir}/vocab.{vocab_type}.pt"
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.min_len = min_len
        self.max_len = max_len
        self.min_kb_len = min_kb_len
        self.max_kb_len = max_kb_len
        self.share_vocab = share_vocab

        self.data = {}
        self.SRC = TextField(tokenize_fn=tokenize, embed_file=embed_file)
        if self.share_vocab:
            self.TGT = self.SRC
            self.KB = self.SRC
        else:
            self.TGT = TextField(tokenize_fn=tokenize, embed_file=embed_file)
            self.KB = TextField(tokenize_fn=tokenize, embed_file=embed_file)

        self.fields = {'src': self.SRC,
                       'tgt': self.TGT,
                       'kb': self.KB}
        
        self.max_lens = {'src': 2*self.max_len, 'tgt': self.max_len, 'kb': self.max_kb_len}

        '''def src_filter_pred(src, t):
            return min_len <= len(self.SRC.tokenize_fn(src)) <= max_len*t

        def tgt_filter_pred(tgt, t):
            return min_len <= len(self.TGT.tokenize_fn(tgt)) <= max_len*t

        def kb_filter_pred(kb, t):
            return min_kb_len <= len(self.KB.tokenize_fn(kb)) <= max_kb_len*t

        self.filter_pred = lambda ex: src_filter_pred(" ".join(ex['src']), ex['turns']) and tgt_filter_pred(" ".join(ex['tgt']), ex['turns']) and kb_filter_pred(" ".join(ex['kb']), ex['turns'])'''

        # load vocab or build vocab if not exists
        self.load_vocab()
        self.padding_idx = self.TGT.stoi[self.TGT.pad_token]

    def load_vocab(self):
        """
        load_vocab
        """
        if not os.path.exists(self.prepared_vocab_file):
            print("Building vocab ...")
            train_file = os.path.join(self.data_dir, "train_data.txt")
            valid_file = os.path.join(self.data_dir, "valid_data.txt")
            test_file = os.path.join(self.data_dir, "test_data.txt")
            train_raw = self.read_data(train_file, data_type="train")
            valid_raw = self.read_data(valid_file, data_type="valid")
            test_raw = self.read_data(test_file, data_type="test")
            data_raw = train_raw + valid_raw + test_raw

            vocab_dict = self.build_vocab(data_raw)
            torch.save(vocab_dict, self.prepared_vocab_file)
            print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))
        else:
            print("Loading prepared vocab from {} ...".format(self.prepared_vocab_file))
            vocab_dict = torch.load(self.prepared_vocab_file)

        for name, vocab in vocab_dict.items():
            if name in self.fields:
                self.fields[name].load_vocab(vocab)
        for name, field in self.fields.items():
            if isinstance(field, TextField):
                print("Vocabulary size of fields {}-{}".format(name.upper(), field.vocab_size))

    def build_vocab(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        field_data_dict = {}
        for name in data[0].keys():
            field = self.fields.get(name)
            if isinstance(field, TextField):
                xs = [x[name] for x in data]
                if field not in field_data_dict:
                    field_data_dict[field] = xs
                else:
                    field_data_dict[field] += xs

        vocab_dict = {}
        for name, field in self.fields.items():
            if field in field_data_dict:
                print("Building vocabulary of field {} ...".format(name.upper()))
                if field.vocab_size == 0:
                    field.build_vocab(field_data_dict[field],
                                      min_freq=self.min_freq,
                                      max_size=self.max_vocab_size)
                vocab_dict[name] = field.dump_vocab()
        return vocab_dict

    def load(self):
        """
        load
        """
        if not os.path.exists(self.prepared_data_file):
            self.data = self.build_data()

        else:
            print("Loading prepared data from {} ...".format(self.prepared_data_file))
            self.data = torch.load(self.prepared_data_file)
            print("Number of examples:",
                  " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def build_data(self):
        """
        build
        """
        train_file = os.path.join(self.data_dir, "train_data.txt")
        valid_file = os.path.join(self.data_dir, "valid_data.txt")
        test_file = os.path.join(self.data_dir, "test_data.txt")

        print("Reading data ...")
        train_raw = self.read_data(train_file, data_type="train")
        valid_raw = self.read_data(valid_file, data_type="valid")
        test_raw = self.read_data(test_file, data_type="test")

        print("Building TRAIN examples ...")
        train_data = self.build_examples(train_raw, data_type="train")
        print("Building VALID examples ...")
        valid_data = self.build_examples(valid_raw, data_type="valid")
        print("Building TEST examples ...")
        test_data = self.build_examples(test_raw, data_type="test")

        '''for i in train_data:
            print(f"{i['dialog_id']}\n{i['src']}\n{i['tgt']}")'''

        data = {"train": train_data,
                "valid": valid_data,
                "test": test_data}
        torch.save(data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))
        return data

    def read_data(self, data_file, data_type=None):
        """
        read_data
        """
        data = []
        with open(data_file, "r") as fr:
            for line in fr:
                sample = json.loads(line.strip())
                conv_id = sample['conv_id']
                turns = sample['turns']
                text = sample['text']
                kb = sample['kb']
                src = []
                tgt = []
                for t in range(0, len(text), 2):
                    if t == 0:
                        u_sent = text[t]
                        s_sent = text[t + 1]
                    else:
                        u_sent = " ".join([text[t-1], text[t]])
                        s_sent = text[t+1]
                    src.append(u_sent)
                    tgt.append(s_sent)
                assert len(src) == turns
                data_sample = {'conv_id': conv_id,
                               'turns': turns,
                               'src': src,
                               'tgt': tgt,
                               'kb': kb,
                               }
                data.append(data_sample)
        print("Read {} {} examples".format(len(data), data_type.upper()))
        return data

    def build_examples(self, data, data_type=None):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        examples = []
        for raw_data in tqdm(data):
            example = {}
            for name, strings in raw_data.items():
                if name in self.fields.keys():
                    if data_type!='test':
                        xx = self.fields[name].numericalize(strings, self.max_lens[name])
                    else:
                        xx = self.fields[name].numericalize(strings, int(1e7))
                    example[name] = xx
                else:
                    example[name] = strings
            examples.append(example)

        return examples

    def create_batches(self, batch_size, data_type="train", shuffle=False):
        """
        create_batches
        """
        try:
            data = self.data[data_type]
            dialog_batcher = DialogBatcher(batch_size=batch_size,
                                           data_type=data_type,
                                           shuffle=shuffle)
            dialog_batcher.prepare_input_list(input_data_list=data)
            return dialog_batcher
        except KeyError:
            raise KeyError("Unsupported data type: {}!".format(data_type))
