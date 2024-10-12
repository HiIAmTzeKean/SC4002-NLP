import json
import os
import datasets
from datasets import Dataset
import nltk

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