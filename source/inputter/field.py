import re
import nltk
import torch
from tqdm import tqdm
from collections import Counter
from source.utils.tokenizer import Tokenizer
from pytorch_pretrained_bert import BertTokenizer


PAD = "[PAD]"
UNK = "[UNK]"
BOS = "[CLS]"
EOS = "[SEP]"
NUM = "<num>"

kbt_tokenizer = Tokenizer('spacy')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', \
                                              never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[END]"))

def tokenize(s):
    """
    tokenize
    """
    
    toks = tokenizer.tokenize(s)
    return toks

def kbt_tokenize(s):
    """
    tokenize
    """
    return kbt_tokenizer(s)

class Field(object):
    """
    Field
    """
    def __init__(self, sequential=False, dtype=None, fix_length=50):
        self.sequential = sequential
        self.dtype = dtype if dtype is not None else int

    def str2num(self, string):
        """
        str2num
        """
        raise NotImplementedError

    def num2str(self, number):
        """
        num2str
        """
        raise NotImplementedError

    def numericalize(self, strings, max_len):
        """
        numericalize
        """
        if isinstance(strings, str):
            return self.str2num(strings, max_len)
        else:
            return [self.numericalize(s, max_len) for s in strings]

    def denumericalize(self, numbers):
        """
        denumericalize
        """
        if isinstance(numbers, torch.Tensor):
            with torch.cuda.device_of(numbers):
                numbers = numbers.tolist()
        if self.sequential:
            if not isinstance(numbers[0], list):
                return self.num2str(numbers)
            else:
                return [self.denumericalize(x) for x in numbers]
        else:
            if not isinstance(numbers, list):
                return self.num2str(numbers)
            else:
                return [self.denumericalize(x) for x in numbers]


class TextField(Field):
    """
    TextField
    """
    def __init__(self,
                 tokenize_fn=None,
                 pad_token=PAD,
                 unk_token=UNK,
                 bos_token=BOS,
                 eos_token=EOS,
                 special_tokens=None,
                 embed_file=None,
                 max_len=50):
        super(TextField, self).__init__(sequential=True, dtype=int, fix_length=max_len)

        self.tokenize_fn = tokenize_fn if tokenize_fn is not None else str.split
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.embed_file = embed_file

        specials = [self.pad_token, self.unk_token,
                    self.bos_token, self.eos_token]
        self.specials = [x for x in specials if x is not None]

        if special_tokens is not None:
            for token in special_tokens:
                if token not in self.specials:
                    self.specials.append(token)

        self.itos = []
        self.stoi = {}
        self.vocab_size = 0
        self.embeddings = None

    def build_vocab(self, texts, min_freq=0, max_size=None):
        """
        build_vocab
        """
        def flatten(xs):
            """
            flatten
            """
            flat_xs = []
            for x in xs:
                if isinstance(x, str):
                    flat_xs.append(x)
                elif isinstance(x, list):
                    for xi in x:
                        flat_xs.append(xi)
                else:
                    raise ValueError("Format of texts is wrong!")
            return flat_xs

        # flatten texts
        texts = flatten(texts)

        counter = Counter()
        for string in tqdm(texts):
            tokens = self.tokenize_fn(string)
            counter.update(tokens)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in self.specials:
            del counter[tok]

        self.itos = list(self.specials)

        if max_size is not None:
            max_size = max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        cover = 0
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
            cover += freq
        cover = cover / sum(freq for _, freq in words_and_frequencies)
        print(
            "Built vocabulary of size {} (coverage: {:.3f})".format(len(self.itos), cover))

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

        if self.embed_file is not None:
            self.embeddings = self.build_word_embeddings(self.embed_file)

    def build_word_embeddings(self, embed_file):
        """
        build_word_embeddings
        """
        if isinstance(embed_file, list):
            embeds = [self.build_word_embeddings(e_file)
                      for e_file in embed_file]
        elif isinstance(embed_file, dict):
            embeds = {e_name: self.build_word_embeddings(e_file)
                      for e_name, e_file in embed_file.items()}
        else:
            cover = 0
            print("Building word embeddings from '{}' ...".format(embed_file))
            with open(embed_file, "r") as f:
                #num, dim = map(str, f.readline().strip().split())
                num, dim = 0, 300
                embeds = [[0] * dim] * len(self.stoi)
                for line in f:
                    w, vs = line.rstrip().split(maxsplit=1)
                    if w in self.stoi:
                        try:
                            vs = [float(x) for x in vs.split(" ")]
                        except Exception:
                            vs = []
                        if len(vs) == dim:
                            embeds[self.stoi[w]] = vs
                            cover += 1
            rate = cover / len(embeds)
            print("{} words have pretrained {}-D word embeddings (coverage: {:.3f})".format( \
                    cover, dim, rate))
        return embeds

    def dump_vocab(self):
        """
        dump_vocab
        """
        vocab = {"itos": self.itos,
                 "stoi": {tok: i for i, tok in enumerate(self.itos)},
                 "embeddings": self.embeddings}
        return vocab

    def load_vocab(self, vocab):
        """
        load_vocab
        """
        self.itos = vocab["itos"]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self.embeddings = vocab["embeddings"]

    def str2num(self, string, max_len):
        """
        str2num
        """
        tokens = []
        unk_idx = self.stoi[self.unk_token]

        if self.bos_token:
            tokens.append(self.bos_token)

        temp = self.tokenize_fn(string)
        if(len(temp) > max_len):
            print(f"trunc: {len(temp)}")
            temp = temp[-max_len:]
        tokens += temp

        if self.eos_token:
            tokens.append(self.eos_token)
        #print(tokens)
        indices = [self.stoi.get(tok, unk_idx) for tok in tokens]
        return indices

    def num2str(self, number):
        """
        num2str
        """
        tokens = [self.itos[x] for x in number]
        if tokens[0] == self.bos_token:
            tokens = tokens[1:]
        text = []
        for w in tokens:
            if w != self.eos_token:
                text.append(w)
            else:
                break
        text = [w for w in text if w not in (self.pad_token, )]
        text = " ".join(text)
        return text