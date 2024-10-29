import json
import os
# import datasets
from datasets import load_dataset
from datasets import Dataset
import nltk
import torch 
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data

UNK_TOKEN = "<UNK>"
EMBEDDING_DIM = 100 # glove embedding are usually 50,100,200,300
SAVE_DIR = './result/'
VOCAB_PATH = os.path.join(SAVE_DIR, 'vocab.json')
EMBEDDING_MATRIX_PATH = os.path.join(SAVE_DIR, 'embedding_matrix.npy')
WORD2IDX_PATH = os.path.join(SAVE_DIR, 'word2idx.json')
IDX2WORD_PATH = os.path.join(SAVE_DIR, 'idx2word.json')

def tokenize(dataset: Dataset, save=False) -> set:
    """Tokenize the text in the dataset using NTLK

    :param dataset: The dataset to tokenize
    :type dataset: Dataset
    :return: The set of tokens in the dataset
    :rtype: set
    """
    vocab = set()
    
    for example in dataset:
        tokens = nltk.word_tokenize(example['text'])
        vocab.update(tokens)
    
    print(f"Vocabulary Size: {len(vocab)}")
    if save:
        with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
            json.dump(list(vocab), f, ensure_ascii=False, indent=4)

        print(f"Vocabulary saved to {VOCAB_PATH}")
    return vocab

def set_seed(seed = 0):
    '''
    set random seed
    '''
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
    word_embedding_glove = word_embedding_glove['train']
    
    for example in word_embedding_glove:
        split_line = example["text"].strip().split()
        word = split_line[0]
        vector = np.array(split_line[1:], dtype='float32')
        glove_dict[word] = vector

    print(f"Total GloVe words loaded: {len(glove_dict)}")
    return glove_dict

class EmbeddingMatrix():
    def __init__(self) -> None:
        self.d = 0 
        self.v = 0
        self.embedding_matrix:np.ndarray
        self.word2idx:dict
    @classmethod
    def load() -> "EmbeddingMatrix":
        # load vectors from file
        embedding_matrix:np.ndarray = np.load(EMBEDDING_MATRIX_PATH)
        # set attributes
        em = EmbeddingMatrix()
        em.embedding_matrix = embedding_matrix
        
        with open(WORD2IDX_PATH, 'r', encoding='utf-8') as f:
            word2idx:dict = json.load(f)
            em.word2idx = word2idx
        em.v, em.d = embedding_matrix.shape
        return em
    @property
    def dimension(self) -> int:
        return self.d
    @property
    def vocab_size(self) -> int:
        return self.v
    @property
    def vocab(self) -> set[str]:
        return set(self.word2idx.keys())
    def __getitem__(self, word:str) -> np.ndarray:
        return self.embedding_matrix[self.word2idx[word]]


class CustomDatasetPreparer:
    def __init__(self, dataset_name, word2idx, unk_token, max_len=512, batch_size=50):
        """
        Initialize the dataset preparer.
        
        :param dataset_name: Name of the dataset to load (e.g., "rotten_tomatoes").
        :param word2idx: Dictionary mapping words to their corresponding IDs.
        :param unk_token: Token for unknown words.
        :param max_len: Maximum sequence length for tokenization.
        :param batch_size: Batch size for DataLoader.
        """
        self.dataset = load_dataset(dataset_name)
        self.word2idx = word2idx
        self.unk_token = unk_token
        self.max_len = max_len
        self.batch_size = batch_size

    def tokenize(self, texts):
        """
        Tokenize the given texts and truncate to max length.
        
        :param texts: List of texts to tokenize.
        :return: Tokenized and padded sequences as tensors and their lengths.
        """
        tokenized = []
        lengths = []
        for text in texts:
            tokens = nltk.word_tokenize(text.lower())
            token_ids = [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in tokens]
            token_ids = token_ids[:self.max_len]  
            tokenized.append(torch.tensor(token_ids, dtype=torch.long))
            lengths.append(len(token_ids))
        return tokenized, lengths

    def prepare_dataset(self, dataset_split, shuffle=False):
        """
        Prepare the dataset for a specific split (train, validation, test).
        
        :param dataset_split: The split of the dataset to process.
        :param shuffle: Whether to shuffle the dataset (default: False).
        :return: DataLoader for the given dataset split.
        """
        set_tokenized, lengths = self.tokenize(dataset_split['text'])

        set_tokenized = pad_sequence(set_tokenized, batch_first=True)

        lengths = torch.tensor(lengths, dtype=torch.long)

        set_labels = torch.tensor(dataset_split['label'], dtype=torch.long)

        extra_features = torch.zeros((len(set_labels), 0), dtype=torch.float)

        sorted_indices = torch.argsort(lengths, descending=True)
        set_tokenized_sorted = set_tokenized[sorted_indices]
        set_labels_sorted = set_labels[sorted_indices]
        extra_features_sorted = extra_features[sorted_indices]
        lengths_sorted = lengths[sorted_indices]

        set_data = data.TensorDataset(set_tokenized_sorted, extra_features_sorted, lengths_sorted, set_labels_sorted)

        return data.DataLoader(set_data, batch_size=self.batch_size, shuffle=shuffle)

    def get_dataloaders(self):
        """
        Prepare and return DataLoaders for the train, validation, and test sets.
        
        :return: A tuple containing train_loader, validation_loader, and test_loader.
        """
        train_loader = self.prepare_dataset(self.dataset['train'], shuffle=False)
        val_loader = self.prepare_dataset(self.dataset['validation'], shuffle=False)
        test_loader = self.prepare_dataset(self.dataset['test'], shuffle=False)
        return train_loader, val_loader, test_loader


