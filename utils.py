from datasets import Dataset
import pickle
import pandas as pd
import numpy as np




TWITTER_DIR = "twitter-datasets/"

def read_twitter_file(path):
    with open(path) as tf:
        tweets = tf.read().split("\n")
    return tweets


def create_datasets(sub_sampling=None):
    tnf = read_twitter_file(TWITTER_DIR + "train_neg_full.txt")
    tnf_labels = np.zeros(len(tnf))
    tpf = read_twitter_file(TWITTER_DIR + "train_pos_full.txt")
    tpf_labels = np.ones(len(tnf))

    ds = Dataset.from_pandas(pd.concat([
        pd.DataFrame({'tweet': tnf, 'labels': tnf_labels}),
        pd.DataFrame({'tweet': tpf, 'labels': tpf_labels}),

    ]).set_index("tweet")).shuffle()

    if(sub_sampling is not None):
        ds = Dataset.from_dict(ds[:sub_sampling])
    
    return ds

