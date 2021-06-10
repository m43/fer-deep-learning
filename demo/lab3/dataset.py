import csv
import os
from abc import ABC, abstractmethod
from collections import namedtuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils.util import project_path

SST_DATASET_DIR = os.path.join(project_path, "datasets/sst")
SST_TRAIN_CSV_FILENAME = "sst_train_raw.csv"
SST_VALID_CSV_FILENAME = "sst_valid_raw.csv"
SST_TEST_CSV_FILENAME = "sst_test_raw.csv"
SST_GLOVE_TXT_FILENAME = "sst_glove_6b_300d.txt"

Instance = namedtuple("Instance", ["text", "sentiment"])


class Vocab(ABC):
    @abstractmethod
    def encode(self, x):
        pass


class SentimentLabelVocab(Vocab):
    sentiment_to_label = {"positive": 0, "negative": 1}

    def encode(self, instance_label):
        return torch.tensor(self.sentiment_to_label[instance_label])


class TokenVocab(Vocab):
    pad_token = "<PAD>"
    unk_token = "<UNK>"

    def __init__(self, token_freq_dict, max_size=-1, min_freq=0):
        self.stoi = {TokenVocab.pad_token: 0, TokenVocab.unk_token: 1}
        assert max_size == -1 or len(self.stoi) < max_size

        for token, freq in sorted(token_freq_dict.items(), key=lambda x: (-x[1])):  # , x[0]
            if freq >= min_freq:
                self.stoi[token] = len(self.stoi)
                if max_size != -1 and len(self.stoi) == max_size:
                    break

        self.itos = {i: s for s, i in self.stoi.items()}

    def encode_token(self, token):
        key = token if token in self.stoi else TokenVocab.unk_token
        return torch.tensor(self.stoi[key])

    def encode(self, sentence):
        return torch.tensor([self.encode_token(token) for token in sentence])

    def get_pad_idx(self):
        return self.stoi[TokenVocab.pad_token]

    def get_unk_idx(self):
        return self.stoi[TokenVocab.unk_token]


class NLPDataset(Dataset):
    def __init__(self, instances, max_vocab_size, min_token_freq_in_vocab, vocab=None):
        self.instances = instances
        if vocab is None:
            token_freq, _ = self.get_frequencies()
            self._text_vocab = TokenVocab(token_freq, max_size=max_vocab_size, min_freq=min_token_freq_in_vocab)
            self.label_vocab = SentimentLabelVocab()
        else:
            self._text_vocab, self.label_vocab = vocab

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        tokens, labels = self.instances[idx]
        return self.text_vocab.encode(tokens), self.label_vocab.encode(labels)

    @property
    def text_vocab(self):
        return self._text_vocab

    @text_vocab.setter
    def text_vocab(self, value):
        # no caching is done, nothing needs to be updated
        self._text_vocab = value

    def get_frequencies(self):
        t_freq, s_freq = {}, {}
        for tokens, sentiment in self.instances:
            for t in tokens:
                t_freq[t] = t_freq.get(t, 0) + 1
            s_freq[sentiment] = s_freq.get(sentiment, 0) + 1
        return t_freq, s_freq

    @staticmethod
    def from_file(path, vocab=None, max_vocab_size=-1, min_token_freq_in_vocab=0):
        ds = pd.read_csv(path, sep=",", header=None, quoting=csv.QUOTE_NONE)
        data = [Instance(t.split(" "), s.strip()) for t, s in zip(ds[0], ds[1])]
        return NLPDataset(data, max_vocab_size, min_token_freq_in_vocab, vocab)

    @staticmethod
    def pad_collate_fn(batch, pad_idx=0):
        texts, labels = zip(*batch)
        lengths = torch.tensor([len(text) for text in texts])
        texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_idx)
        return texts, torch.tensor(labels, dtype=torch.float), lengths


def load_sst_dataset(max_vocab_size=-1, min_token_freq_in_vocab=0):
    train = NLPDataset.from_file(os.path.join(SST_DATASET_DIR, SST_TRAIN_CSV_FILENAME), None, max_vocab_size,
                                 min_token_freq_in_vocab)
    vocab = train.text_vocab, train.label_vocab
    valid = NLPDataset.from_file(os.path.join(SST_DATASET_DIR, SST_VALID_CSV_FILENAME), vocab)
    test = NLPDataset.from_file(os.path.join(SST_DATASET_DIR, SST_TEST_CSV_FILENAME), vocab)
    test.text_vocab = valid.text_vocab = train.text_vocab
    return train, valid, test


class Word2Vec:
    @staticmethod
    def generate_random_gauss_matrix(strings, pad_idx=0, dim=300):
        word2vec = torch.randn((len(strings), dim))
        word2vec[pad_idx] = torch.zeros(dim)
        return torch.nn.Embedding.from_pretrained(word2vec)

    @staticmethod
    def load_glove(strings, pad_idx=0, path=os.path.join(SST_DATASET_DIR, SST_GLOVE_TXT_FILENAME)):
        glove = pd.read_table(path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        dim = glove.shape[1]

        word2vec = torch.zeros((len(strings), dim), dtype=torch.float32)
        for i, string in enumerate(strings):
            word2vec[i] = torch.tensor(glove.loc[string].to_numpy()) if string in glove.index else torch.zeros(dim)
        word2vec[pad_idx] = torch.zeros(dim)

        return torch.nn.Embedding.from_pretrained(word2vec)


if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset = load_sst_dataset()
    batch_size = 2
    shuffle = False
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, collate_fn=NLPDataset.pad_collate_fn)
    texts, labels, lengths = next(iter(train_dataloader))
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")

    token_freq, label_freq = train_dataset.get_frequencies()
    assert token_freq["the"] == 5954
    assert token_freq["a"] == 4361
    assert token_freq["and"] == 3831
    assert token_freq["of"] == 3631
    assert token_freq["to"] == 2438

    assert len(train_dataset.text_vocab.itos) == 14806
    assert train_dataset.text_vocab.stoi["the"] == 2
    assert train_dataset.text_vocab.stoi["a"] == 3
    assert train_dataset.text_vocab.stoi["and"] == 4
    assert train_dataset.text_vocab.stoi["my"] == 188

    print(label_freq)
    instance_text, instance_label = train_dataset.instances[3]
    print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")
    print(f"Numericalized text: {train_dataset.text_vocab.encode(instance_text)}")
    print(f"Numericalized label: {train_dataset.label_vocab.encode(instance_label)}")

    ordered_vocab_tokens = [token for _, token in sorted(train_dataset.text_vocab.itos.items())]
    pad_idx = train_dataset.text_vocab.get_pad_idx()
    word2vec = Word2Vec.generate_random_gauss_matrix(ordered_vocab_tokens, pad_idx, 300)
    assert word2vec.weight.shape[0] == len(train_dataset.text_vocab.stoi.keys())
    assert word2vec.weight.shape[1] == 300

    word2vec = Word2Vec.load_glove(ordered_vocab_tokens, pad_idx)
    assert word2vec.weight.shape[0] == len(train_dataset.text_vocab.stoi.keys())
    assert word2vec.weight.shape[1] == 300
    assert word2vec(train_dataset.text_vocab.encode_token("braveheart"))[0] == 0.2438
    assert word2vec(train_dataset.text_vocab.encode_token("braveheart"))[1] == -0.21141
    assert word2vec(train_dataset.text_vocab.encode_token("braveheart"))[-1] == 0.059403
