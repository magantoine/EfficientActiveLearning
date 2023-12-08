from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
import pickle
import pandas as pd
import numpy as np
import torch




TWITTER_DIR = "data/twitter-datasets/"

def read_twitter_file(path):
    with open(path) as tf:
        tweets = tf.read().split("\n")
    return tweets


def create_datasets(sub_sampling=None, return_type="ds"):
    assert(return_type in ["ds", "df"], "Return type either 'df' for DatFrame or 'ds' for Dataset")
    tnf = read_twitter_file(TWITTER_DIR + "train_neg_full.txt")
    tnf_labels = np.zeros(len(tnf))
    tpf = read_twitter_file(TWITTER_DIR + "train_pos_full.txt")
    tpf_labels = np.ones(len(tnf))

    
    ds = HFDataset.from_pandas(pd.concat([
        pd.DataFrame({'tweet': tnf, 'labels': tnf_labels}),
        pd.DataFrame({'tweet': tpf, 'labels': tpf_labels}),
        ]
    ).set_index("tweet")).shuffle()

    if(sub_sampling is not None):
        ds = ds[:sub_sampling]

    cast_class = HFDataset if(return_type == 'ds') else pd.DataFrame
    
    return cast_class.from_dict(ds)




class MyDataset(Dataset):
    """
        Dataset class that process the samples and puts them on the necessary device
    """

    def __init__(self, data, tokenizer, max_length, device) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.initial_data = data ## used for introducing of randomness

    def __len__(self):
        # Return the total number of samples in your dataset
        return len(self.data)


    def __getitem__(self, idx):
        # Retrieve and preprocess a single sample at the given index

        sample = self.data[idx]

        # Use the tokenizer to tokenize the input text
        inputs = self.tokenizer(
            sample["tweet"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # You might want to include other information such as labels

        labels = sample.get("labels", [])

        # Return a dictionary containing the input_ids, attention_mask, and labels
        return {
            "input_ids": inputs["input_ids"].squeeze(0).to(self.device),
            "attention_mask": inputs["attention_mask"].squeeze(0).to(self.device),
            "labels": torch.tensor(labels).to(self.device),
        }


AS_TYPES = ["unif_140", "unif_200", "entr_140", "entr_200"]
SPLITS = ["train", "val"]
DATASET_TYPES = ["ds", "df"]
def load_aware_sampling(as_type=AS_TYPES[0], split="train", return_type=DATASET_TYPES[0]):
    
    if(as_type not in AS_TYPES):
        raise ValueError(f"Aware sampling type must be one of {AS_TYPES}")
    if(split not in SPLITS):
        raise ValueError(f"Split must be one of {SPLITS}")
    if(return_type not in DATASET_TYPES):
        raise ValueError(f"Dataset type must be one of {DATASET_TYPES}")
    
    with open(f"./data/twitter-datasets/aware_sampling_{as_type}_{split}.pkl", 'rb') as f:
        df = pickle.load(f)

    df = df[["tweet", "label"]]
    df.columns = ["tweet", "labels"]
    df['labels'] = df["labels"].apply(lambda l : 0 if l == -1 else 1)

    if(return_type == "df"):
        return df
    elif(return_type == "ds"):
        return HFDataset.from_pandas(df)
        