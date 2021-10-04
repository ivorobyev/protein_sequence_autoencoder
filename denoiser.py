import numpy as np 
from model import cnn_model_builder, lstm_model_builder
from sklearn.preprocessing import OneHotEncoder
import math
import random

def reconstruct_sequence(seq, model):
    off = 60
    probs = [seq[a:a+off] for a in range(len(seq)-off+1)]
    a = np.array(probs)
    preds_s = model.predict(a)
    diags = [preds_s[::-1,:].diagonal(i) for i in range(-preds_s.shape[0]+1,preds_s.shape[1])]
    return np.array([np.mean(a, axis=1) for a in diags])
        
def prepare_sequence(seq):
    categories_ = ['A','C','D','E','F','G','H',
                   'I','K','L','M','N','P','Q',
                   'R','S','T','V','W','Y']

    encoder = OneHotEncoder(categories = [categories_])
    seq = encoder.fit_transform(np.array(list(seq)).reshape(-1, 1)).toarray()
    pos = [random.randint(1, len(seq)-1) for a in range(math.ceil(len(seq)*0.1))]
    for a in pos:
        seq[a] = [0]*len(categories_)
    return seq, pos