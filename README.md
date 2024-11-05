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
the existing pretrained word embeddings. The system is used to perfor a sentence classification
task. 

In this project, we will implement the following model:
- Recurrent Neural Network (RNN)
- Bidirectional LSTM (BiLSTM)
- Bidirectional LSTM with Attention
- Bidirectional GRU (BiGRU)
- Bidirectional GRU with Attention
- Convolutional Neural Network (CNN)

## Installation and Setup

### 1. Clone the Repository
```
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

## Usage
This project is organised into different Jupyter notebooks located in the codebase. 
Each notebook will corresponds to a specific part of the assignment. 

### 1. common_utils.py
This python script will contain specific common utilities function. 

### 2. Preparing Word Embedding (part_1.ipynb)
This notebook focuses on preparing the word embeddings to form the input layer of the 
model. 

### 3. Model Training & Evaluation (RNN) (part_2.ipynb)
This notebook focuses on implementation of Recurrent Neural Network as well as the implementation
of a variety of RNN using multiple techniques to increase the accuracy score. 

### 4. Enhancement (Part 3.1 and Part 3.2) (part_3.ipynb)
Implementing various enhancement, focused on updating the word embeddings during the training process
as well as applying the solution to mitigate OOV words. 

### 5. Enhancement (BiGRU Model) (part_3_BiGRU.ipynb)
Implementation of BiGRU model and analysing its hyperparameter. 

### 6. Enhancement (BiLSTM Naive Model) (part_3_bilstm.ipynb)
Implementation of BiLSTM Naive Model and analysing its hyperparameter. 

### 7. Enhancement (BiLSTM and BiGRU with Attention) (part_3_GRU_LSTM.ipynb)
Implementation of both BiLSTM and BiGRU with attention to refine and test out its accuracy with attention. 

### 8. Enhancement (CNN) (part_3-wy.ipynb)
Implementation of CNN, Dual Channel CNN as well as researching on other optimisations. 

To Do WY:
1. Do HyperParameter Training for CNN
2. ResNet Archi
3. Length Based Curriculum
