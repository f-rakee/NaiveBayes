import glob
import os
import numpy as np
from collections import Counter

# Nc: Number of document from D in class c
Nc = [0, 0]
# Classes: 0: negative class, 1: positive class
C = [0, 1]


# A function to read files:
def read_files(address):
    file_list = glob.glob(os.path.join(os.getcwd(), address, "*.txt"))

    temp_list = []
    for file_path in file_list:
        with open(file_path) as f_input:
            temp_list.append(f_input.read())
    return temp_list


# ***************** Train Phase ******************************

print("Start Train phase ... ")
#  Read Negative Train data
train_neg_corpus = read_files('neg_train files')
Nneg_train = len(train_neg_corpus)
# print("train neg", train_neg_corpus)

# Read Positive Train data
train_pos_corpus = read_files('pos_train files')
Npos_train = len(train_pos_corpus)
# print("train pos", train_pos_corpus)

# D: all train documents
D = train_neg_corpus + train_pos_corpus
Ndoc = len(D)
Nc = [Nneg_train, Npos_train]

# Find Vocabulary of Negative Train data:
V_neg = []
for item in train_neg_corpus:
    V_neg = item.split() + V_neg

# Find Vocabulary of Positive Train data:
V_pos = []
for item in train_pos_corpus:
    V_pos = item.split() + V_pos

V = V_neg + V_pos

# Nv: Total number of vocabularies
Nv = len(V)

# Nv_each_class: Number of vocabularies in each class
Nv_each_class = [len(V_neg), len(V_pos)]

# Save frequency of each word in each class
count = {}
for i in range(len(C)):
    if i == 0:
        count[i] = Counter(V_neg)
    else:
        count[i] = Counter(V_pos)


def TrainNaiveBayes(D, C):
    loglikelihood = {}
    logprior = np.zeros((len(C), 1))

    for c in range(len(C)):
        logprior[c] = np.log(Nc[c]/Ndoc)
        loglikelihood[c] = Counter()
        for w in V:
            loglikelihood[c][w] = np.log((count[c][w] + 1) / (Nv_each_class[c] + Nv))

    return logprior, loglikelihood, V


logprior, loglikelihood, V = TrainNaiveBayes(D, C)

# *********************** Test ****************************
print("Start Test phase ... ")
# Read Negative Test data
test_neg_corpus = read_files('neg_test files')
Nneg_test = len(test_neg_corpus)

# Read Positive Test data
test_pos_corpus = read_files('pos_test files')
Npos_test = len(test_pos_corpus)

data_test = {}
for i in range(len(C)):
    if i == 0:
        data_test[i] = test_neg_corpus
    else:
        data_test[i] = test_pos_corpus

Ndoc_test = Npos_test + Nneg_test
sumc = np.zeros((2, 1))

def TestNaiveBayes(test_doc, logprior, loglikelihood, V):
    for c in range(len(C)):
        sumc[c] = logprior[c]
        for word in test_doc.split():
            if word in V:
                sumc[c] = sumc[c] + loglikelihood[c][word]

    max_idx = np.argmax(sumc)

    return max_idx


# ********************** Calculate Accuracy *******************
num_true = 0
for i in range(len(C)):
    for j in range(len(data_test[i])):
        predicted_class = TestNaiveBayes(data_test[i][j], logprior, loglikelihood, V)
        if predicted_class == i:
            num_true += 1

