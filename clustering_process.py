from transformers import (AutoModel, AutoTokenizer)
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sns
import numpy as np
from utils import MyDataset
import torch
from sklearn.metrics.pairwise import (cosine_similarity,
                                      euclidean_distances)
import pickle
import os


clusters = np.loadtxt("clustering_center.npy")
CHUNK_SIZE = 10_000
full_ds_file = open('./data/twitter-datasets/train_pos_full.txt').readlines()
N_SAMPLES = len(full_ds_file)
print(len(full_ds_file))
cur_clusters = []

# mps_device = "cuda"
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print("MPS connected")
else:
    raise ImportError("No MPS found")


EMB_MODEL = "vinai/bertweet-base"

emb_tok = AutoTokenizer.from_pretrained(EMB_MODEL)
emb_model = AutoModel.from_pretrained(EMB_MODEL).to(mps_device)


def check_current_evals(fname="pos1_clusters.pkl"):
  if(os.path.exists(fname)):
    with open(fname, 'rb') as f:
      return pickle.load(f)
  else :
    return []


## first split
start = len(check_current_evals("pos1_clusters.pkl"))
start += len(check_current_evals("pos2_clusters.pkl"))
start += len(check_current_evals("pos3_clusters.pkl"))

def get_chunk_emb(data):
    ds = [
        {
            "tweet": st,
            "labels": 0 ## who cares
        } for st in data
    ]
    ds = MyDataset(data=ds,
          tokenizer=emb_tok,
          max_length=130,
          device=mps_device)


    process_dl = DataLoader(ds,
                            batch_size=32,
                            shuffle=False)

    process_embs = []
    for batch in process_dl:
        batch.pop("labels")
        try : 
          process_embs.append(
              emb_model(
                      **batch
                  ).last_hidden_state[:, 0, :].detach()
          )
        except:
          print("Failed")
          ## a zeros batch
          process_embs.append(torch.zeros((32, 768)))
          continue
    return process_embs




print("STARTING AT ", start)
for i in tqdm(range(start, N_SAMPLES, CHUNK_SIZE)): ## all the samples
    data = full_ds_file[i: min(i+CHUNK_SIZE, N_SAMPLES)]

    all_embs = torch.cat(
        get_chunk_emb(data)
    ).to("cpu").numpy()

    chunk_dists = euclidean_distances(all_embs, clusters).argmin(axis=1)
    chunk_cosine = cosine_similarity(all_embs, clusters).argmax(axis=1)

    cur_clusters = zip(
        np.arange(i, i + CHUNK_SIZE),

        chunk_dists,
        chunk_cosine,
        [x for x in all_embs]
    )

    done_evals = list(check_current_evals("pos3_clusters.pkl"))
    with open("pos3_clusters.pkl", "wb") as f:
        pickle.dump(done_evals + list(cur_clusters),f)
    cur_clusters = []