# SC4002 Natural Language Processing

Nanyang Technological University SC4002 Natural Language Processing Group Project.
This is a group assignment for our module in Sem 1 2024. This repository will 
contain the code solutions for the project and a write up detailing the project.

## Group Members
- Ng Tze Kean
- Ng Woon Yee
- Yap Shen Hwei
- Lim Boon Hien
- Phee Kian Ann
- Lim Ke En

## Project Overview
This project will focus on building a general classification system built on top of
the existing pretrained word embeddings. The system is used to perform sentence
classification task. The dataset that we will be using for is the rotten tomatoes
from the datasets library.

In this project, we will implement the following models:
- Recurrent Neural Network (RNN)
- Bidirectional LSTM (BiLSTM)
- Bidirectional LSTM with Attention
- Bidirectional GRU (BiGRU)
- Bidirectional GRU with Attention
- Convolutional Neural Network (CNN)

## Installation and Setup

### 1. Clone the Repository

```shell
https://github.com/HiIAmTzeKean/SC4002-NLP.git
cd SC4002-NLP
```

### 2. To run the code
To run the code in the notebooks, you will need to install the dependencies. The
dependencies are managed using Poetry. To install the dependencies, run the
following commands:

```shell
pip install poetry
poetry install
```

We suggest that you use a virtual environment to manage the dependencies. To
ensure that there are no conflicting dependencies with your own computer. The
code above will install the virtual environment for you and you would have to
select that virtual environment to run the code in the notebooks.


## Organization
This project is organized into different Jupyter notebooks located in the codebase. 
Each notebook will corresponds to a specific part of the assignment. 

### 1. common_utils.py
This python script will contain specific common utilities function. 

### 2. Preparing Word Embedding (part_1.ipynb)
This notebook focuses on preparing the word embeddings to form the input layer of the 
model. 

### 3. Model Training & Evaluation (RNN) (part_2.ipynb)
This notebook focuses on implementation of Recurrent Neural Network as well as the implementation
of a variety of RNN using multiple techniques to increase the accuracy score. 

### 4. Enhancement (Part 3(a) and Part 3(b)) (part_3_a_b.ipynb)
Implementing various enhancement, focused on updating the word embeddings during the training process
as well as applying the solution to mitigate OOV words. 

### 5. Enhancement (BiLSTM Naive Model) (part_3_c_biLSTM_naive.ipynb)
Implementation of BiLSTM Naive Model and analysing its hyperparameter. 

### 6. Enhancement (BiGRU Model with Attention) (part_3_c_BiGRU.ipynb)
Implementation of BiGRU complex model with attention and analysing its hyperparameter. Extra testing by directly replacing BiGRU layer with BiLSTM to see which is better.

### 7. Enhancement (CNN) (part_3_d_CNN.ipynb)
Implementation of CNN, Dual Channel CNN as well as exploring some parts of the
dataset to figure out how we can better optimize.

### 8. Enhancement (Part 3(e) and Part 3(f)) (part_3_e_f.ipynb)
Here we explore variations of the RNN model with attention and concatenation of
different pooling methods to explore the best model. We also attempted 
self-attention models and negation handling to improve the model.

## Results

We have collated and summarized the results of the models that we have implemented
in the table below. The table will contain the model name, the parameters used
to train the model, the test loss and the test accuracy of the model. Note that
we omitted most of the training loss for Part 2 as the question wanted the validation
accuracy and the test accuracy instead. You can refer to the notebook for
more information.

| Model | Parameters | Test Loss | Test Accuracy |
|---|---|---|---|
| **Part 2: RNN** |  |  |  |
| RNN (Version 1) | Max Pooling on Hidden Layer | 0.547 | 0.745 |
| Optimal RNN | Max pooling, Initial LR: 0.0001, Batch: 32, Adam, Epochs: 33, Dropout: 0.3, Hidden: 128, Layers: 2 | 0.490 | 0.773 |
| **Part 3: RNN** |  |  |  |
| RNN (No Unknown Handling, Unfrozen Embeddings) | Same as Optimal RNN params | 0.509 | 0.775 |
| RNN (Unknown Handling) | Same as Optimal RNN params | 0.595 | 0.783 |
| BiLSTM (Naive) | Max Grad Clip: 5, Layers: 2, Dropout: 0.3, LR: 0.0001, Hidden: 8 | 0.516 | 0.771 |
| BiLSTM (GRU Config) | Max/Avg Pooling, Hidden: 32, Layers: 1, Dropout: 0.5, LR: 0.001, Spatial Dropout | 0.4865 | 0.8039 |
| BiGRU | Max/Avg Pooling, Hidden: 8, Layers: 1, Dropout: 0.5, LR: 0.001, Spatial Dropout | 0.4315 | 0.8011 |
| CNN | Dual Channels, Attention, Filters: 100, Filter Sizes: [3, 5, 7], Dropout: 0.5, LR: 0.001 | 0.4172 | 0.8143 |
| RNN | Max Pooling, Attention, Layers: 1, Dropout: 0.5, Hidden: 100 | 0.419 | 0.814 |
| RNN | Max/Avg Pooling, Concat, Layers: 1, Dropout: 0.5, Hidden: 100 | 0.604 | 0.782 |
| Self-Attention | Layers: 1, Dropout: 0.5, Hidden: 100 | 1.487 | 0.760 |
| Self-Attention | Positional Encoding, Layers: 1, Dropout: 0.5, Hidden: 100 | 0.524 | 0.750 |

## Report

The report for this project can be found in the `report` folder. The report will
contain the details of the project and highlight the key points of the project.