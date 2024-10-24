import json
import os
import datasets
from datasets import Dataset, load_dataset
import nltk

import numpy as np

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
    def load(cls) -> "EmbeddingMatrix":
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