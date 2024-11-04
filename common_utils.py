import json
import os
import random

import datasets
import nltk
import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
HIDDEN_SIZE = 128
NUM_EPOCHS = 100
BATCH_SIZE = 50
LEARNING_RATE = 0.01
EMBEDDING_DIM = 100  # glove embedding are usually 50,100,200,300
SAVE_DIR = "./result/"
VOCAB_PATH = os.path.join(SAVE_DIR, "vocab.json")
EMBEDDING_MATRIX_PATH = os.path.join(SAVE_DIR, "embedding_matrix.npy")
WORD2IDX_PATH = os.path.join(SAVE_DIR, "word2idx.json")
IDX2WORD_PATH = os.path.join(SAVE_DIR, "idx2word.json")


def tokenize(dataset: Dataset, save=False) -> set:
    """Tokenize the text in the dataset using NTLK

    :param dataset: The dataset to tokenize
    :type dataset: Dataset
    :return: The set of tokens in the dataset
    :rtype: set
    """
    vocab = set()

    for example in dataset:
        tokens = nltk.word_tokenize(example["text"])
        vocab.update(tokens)

    print(f"Vocabulary Size: {len(vocab)}")
    if save:
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(list(vocab), f, ensure_ascii=False, indent=4)

        print(f"Vocabulary saved to {VOCAB_PATH}")
    return vocab


def set_seed(seed=0):
    """
    set random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_glove_embeddings() -> dict:
    """Load GloVe embeddings

    :return: GloVe embeddings
    :rtype: Dict
    """
    print("Loading GloVe embeddings...")
    glove_dict = {}
    word_embedding_glove = load_dataset("SLU-CSCI4750/glove.6B.100d.txt")
    word_embedding_glove = word_embedding_glove["train"]

    for example in word_embedding_glove:
        split_line = example["text"].strip().split()
        word = split_line[0]
        vector = np.array(split_line[1:], dtype="float32")
        glove_dict[word] = vector

    print(f"Total GloVe words loaded: {len(glove_dict)}")
    return glove_dict


class EmbeddingMatrix:
    def __init__(self, unk_token=UNK_TOKEN, handle_unknown=True) -> None:
        self.d = 0
        self.v = 0
        self.pad_idx: int
        self.unk_idx: int
        self.embedding_matrix: np.ndarray
        self.word2idx: dict
        self.idx2word: dict
        self.unk_token = unk_token
        self.handle_unknown = handle_unknown

    @classmethod
    def load(cls) -> "EmbeddingMatrix":
        # load vectors from file
        embedding_matrix: np.ndarray = np.load(EMBEDDING_MATRIX_PATH)
        # set attributes
        em = cls()
        em.embedding_matrix = embedding_matrix

        with open(WORD2IDX_PATH, "r", encoding="utf-8") as f:
            word2idx: dict = json.load(f)
            em.word2idx = word2idx
        with open(IDX2WORD_PATH, "r", encoding="utf-8") as f:
            idx2word: dict = json.load(f)
            em.idx2word = idx2word
            
        em.v, em.d = embedding_matrix.shape
        return em

    def save(self) -> None:
        np.save(EMBEDDING_MATRIX_PATH, self.embedding_matrix)
        
        with open(WORD2IDX_PATH, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=4)
            
        with open(IDX2WORD_PATH, "w", encoding="utf-8") as f:
            json.dump(self.idx2word, f, ensure_ascii=False, indent=4)

    @property
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.embedding_matrix, dtype=torch.float)

    def add_padding(self) -> None:
        if "<PAD>" in self.word2idx:
            return
        padding = np.zeros((1, self.d), dtype="float32")
        self.embedding_matrix = np.vstack((self.embedding_matrix, padding))

        self.v += 1
        self.pad_idx = self.v - 1
        self.word2idx["<PAD>"] = self.pad_idx

    def add_unk_token(self) -> None:
        if self.unk_token in self.word2idx:
            return
        unk_vector = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
        self.embedding_matrix = np.vstack((self.embedding_matrix, unk_vector))

        self.v += 1
        self.unk_idx = self.v - 1
        self.word2idx[self.unk_token] = self.unk_idx

    @property
    def dimension(self) -> int:
        """Dimension of the embedding matrix

        :return: The dimension of the embedding matrix
        :rtype: int
        """
        return self.d

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the embedding matrix

        :return: The vocabulary size of the embedding matrix
        :rtype: int
        """
        return self.v

    @property
    def vocab(self) -> set[str]:
        """Vocabulary of the embedding matrix

        Set of words in the embedding matrix

        :return: The vocabulary of the embedding matrix
        :rtype: set[str]
        """
        return set(self.word2idx.keys())

    def __getitem__(self, word: str) -> np.ndarray:
        return self.embedding_matrix[self.word2idx[word]]

    def get_idx(self, word: str) -> int:
        # if word not in vocab, return None
        if self.handle_unknown:
            return self.word2idx.get(word, self.unk_idx)

        return self.word2idx.get(word, None)
    
    def is_in_vocab(self, word: str) -> bool:
        return word in self.word2idx

    def is_in_index(self, idx: int) -> bool:
        return idx in self.idx2word


class EmbeddingsDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        word_embeddings: EmbeddingMatrix,
        sort=False,
        ignore_unknown=True,
        allow_unknown=False,
    ):
        self.word_embeddings = word_embeddings
        tokenized_sentences = []
        self.ignore_unknown = ignore_unknown
        self.allow_unknown = allow_unknown
        for sentence in X:
            tokens = self.tokenize_sentence(sentence)
            tokenized_sentences.append(tokens)

        # Combine tokens, labels, and lengths into a list of tuples
        data = list(zip(tokenized_sentences, y))
        # Sort the data based on the length of the tokenized sentences
        if sort:
            data.sort(
                key=lambda x: len(x[0]), reverse=False
            )  # Set reverse=True for descending order

        # Unzip the sorted data back into tokens and labels
        self.tokens_list, self.labels_list = zip(*data)
        self.len = len(self.tokens_list)

    def __getitem__(self, index):
        # tokenize the sentence
        return self.tokens_list[index], self.labels_list[index]

    def __len__(self):
        return self.len

    def tokenize_sentence(self, x):
        """
        returns a list containing the embeddings of each token
        """
        tokens = nltk.word_tokenize(x)
        # word tokens to index, skip if token is not in the word embeddings
        if self.ignore_unknown:
            tokens = [
                self.word_embeddings.get_idx(token)
                for token in tokens
                if self.word_embeddings.get_idx(token) is not None
            ]
        else:
            tokens = [self.word_embeddings.get_idx(token) for token in tokens]
        return tokens


class CustomDatasetPreparer:
    def __init__(self, dataset_name, batch_size=BATCH_SIZE):
        """
        Initialize the dataset preparer.

        :param dataset_name: Name of the dataset to load (e.g., "rotten_tomatoes").
        :param batch_size: Batch size for DataLoader.
        """
        self.dataset = load_dataset(dataset_name)
        self.batch_size = batch_size
        # word embeddings
        self.word_embeddings = EmbeddingMatrix.load()
        self.word_embeddings.add_padding()
        self.word_embeddings.add_unk_token()

    def load_dataset(self):
        # load dataset from huggingface first
        dataset = load_dataset("rotten_tomatoes")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        test_dataset = dataset["test"]

        train_dataset_ed = EmbeddingsDataset(
            train_dataset["text"],
            train_dataset["label"],
            self.word_embeddings,
            ignore_unknown=False,
        )
        validation_dataset_ed = EmbeddingsDataset(
            validation_dataset["text"],
            validation_dataset["label"],
            self.word_embeddings,
            ignore_unknown=False,
        )
        test_dataset_ed = EmbeddingsDataset(
            test_dataset["text"],
            test_dataset["label"],
            self.word_embeddings,
            ignore_unknown=False,
        )
        return train_dataset_ed, validation_dataset_ed, test_dataset_ed

    def get_dataloaders(self):
        train_dataset_ed, validation_dataset_ed, test_dataset_ed = (
            self.load_dataset()
        )

        def pad_collate(batch, pad_value):
            (xx, yy) = zip(*batch)
            # get the lengths of each sequence
            lengths = [len(x) for x in xx]
            # convert lengths to a tensor
            lengths = torch.tensor(lengths, dtype=torch.long)

            # convert xx to a tensor
            xx = [torch.tensor(x, dtype=torch.int64) for x in xx]
            xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)

            labels = torch.tensor(yy, dtype=torch.long)
            extra_features = torch.zeros((len(labels), 0), dtype=torch.float)

            return xx_pad, extra_features, lengths, labels

        pad_value = self.word_embeddings.pad_idx
        # implement minibatch training
        train_dataloader = DataLoader(
            train_dataset_ed,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: pad_collate(x, pad_value),
        )
        validation_dataloader = DataLoader(
            validation_dataset_ed,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: pad_collate(x, pad_value),
        )
        test_dataloader = DataLoader(
            test_dataset_ed,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: pad_collate(x, pad_value),
        )

        return train_dataloader, validation_dataloader, test_dataloader
