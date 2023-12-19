from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import RAdam
import random

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModelForSequenceClassification,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          AdamW,
                          get_scheduler,
                          Trainer)

import evaluate
accuracy = evaluate.load("accuracy")

# ----------------------------- Global Variables ----------------------------- #

AS_TYPES = ["unif_140", "unif_200", "entr_140", "entr_200"]

# ------------------------------ Dataset Classes ----------------------------- #
    
class ActiveLearningModel(nn.Module):
    def __init__(self, model, tok, teacher):
        super().__init__()
        self.model = model
        self.tok = tok
        self.teacher = teacher

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels=None) -> torch.Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Send to teacher model
        self.teacher.update(output.logits)

        return output


class MyDataset(Dataset):
    def __init__(self, data, tokenizer) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.initial_data = data ## used for introducing of randomness

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Use the tokenizer to tokenize the input text
        inputs = self.tokenizer(
            sample["tweet"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # You might want to include other information such as labels
        labels = sample.get("labels", [])

        # Return a dictionary containing the input_ids, attention_mask, and labels
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(labels),
        }

class TeacherDataset(MyDataset):
    def __init__(self, data, tokenizer, device, T=1_000):
        # self.data = self.load_data()
        super().__init__(data, tokenizer)
        self.T = T
        self.device = device
        self.allHs = torch.tensor([]).to(self.device)

    def update(self, past_logits, automatic_new_iter=False):
        Ps = torch.softmax(past_logits, dim=0)
        nHs = (-Ps * torch.log(Ps)).sum(dim=-1).reshape((-1,))
        if(len(self.allHs) + len(nHs) <= len(self.data)): ## test data
          self.allHs = torch.cat([self.allHs, nHs])
          if(automatic_new_iter and (self.allHs.size()[0] == len(self.data))):
              self.new_iter()

    def new_iter(self, add_randomness=False):
        if(not add_randomness):
          selected_idx = self.allHs.argsort()[-int(len(self.allHs) / 2):] ## take the sample above the entropy median
          print("> TEACHER ROUND, from", len(self.data), "samples to", len(selected_idx))
          self.data = HFDataset.from_dict(self.data[selected_idx.tolist()])

        else :
          CHUNK_SIZE = int(1 * len(self.allHs) / 4)
          high_H_idxs = self.allHs.argsort()[-CHUNK_SIZE:] ## 1/4 of high entropy samples
          high_H_samples = HFDataset.from_dict(self.data[high_H_idxs.tolist()])
          random_init_samples = HFDataset.from_dict(self.data[torch.randperm(len(self.data))[:CHUNK_SIZE]])
          print("> TEACHER ROUND, from", len(self.data), "samples to", len(high_H_idxs) + len(random_init_samples))
          self.data = concatenate_datasets([random_init_samples, high_H_samples])

        self.allHs = torch.tensor([]).to(self.device)

        if(len(self.data) < self.T):
          print(f"Threshold was set at T = {self.T}, {len(self.data)} remaining datapoints, halting.")
          self.data = HFDataset.from_dict({}) ## empty dataset ===> halt

    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
      labels = inputs.get("labels")
      labels = labels.long()
      # forward pass
      outputs = model(input_ids=inputs["input_ids"].squeeze(1), attention_mask=inputs["attention_mask"].squeeze(1))
      logits = outputs.get("logits")
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits, labels.long())
      return (loss, outputs) if return_outputs else loss

# ---------------------------------- Loaders --------------------------------- #
  
def read_twitter_file(path):
    with open(path) as tf:
        tweets = tf.read().split("\n")
    return tweets
   
def create_datasets(sub_sampling=None, data_path="data/twitter-datasets/"):
    tnf = read_twitter_file(data_path + "train_neg_full.txt")
    tpf = read_twitter_file(data_path + "train_pos_full.txt")

    df = pd.concat([
        pd.DataFrame({'tweet': tnf, 'labels': np.zeros(len(tnf))}),
        pd.DataFrame({'tweet': tpf, 'labels': np.ones(len(tnf))}),
    ]).set_index("tweet")

    if(sub_sampling is not None):
        df = df.sample(sub_sampling)

    return HFDataset.from_pandas(df).shuffle()

def load_aware_sampling(data_path, test_ratio=0.3, as_type=AS_TYPES[0]):
    if(as_type not in AS_TYPES):
        raise ValueError(f"Aware sampling type must be one of {AS_TYPES}")
    
    train_df = pd.read_pickle(f"{data_path}/aware_sampling_{as_type}_train.pkl")[["tweet", "label"]]
    train_df.columns = ["tweet", "labels"]
    train_df['labels'] = train_df["labels"].apply(lambda l : 0 if l == -1 else 1)
    train_ds = HFDataset.from_pandas(train_df)

    test_ds = create_datasets(sub_sampling=int(len(train_ds)*test_ratio))

    return {'train': train_ds, 'test': test_ds}

def tokenize(ds, tokenizer):
   tokenized = ds.map(lambda x : tokenizer(x["tweet"], return_tensors="pt", truncation=True, padding='max_length'), batched=True)
   tokenized = tokenized.remove_columns(["tweet"])
   return tokenized

def load_data(data_path, N, model_name, test_ratio=.3, active_learning=False, T=10_000, aware_sampling=False, aware_sampling_type='unif_140', device='cuda:0'):
    """
        Load, split and tokenizes the tweet dataset
    """
    # Create dataset
    print('Creating the dataset...')
    ds = create_datasets(sub_sampling=N, data_path=data_path).train_test_split(test_size=test_ratio) if aware_sampling == False \
                             else load_aware_sampling(data_path, test_ratio=test_ratio, as_type=aware_sampling_type)

    # Get the model's tokenizer.
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if "gpt2" in model_name:
        # default to left padding
        tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token

    print('Preprocessing the Training Data...')
    train_ds = tokenize(ds['train'], tokenizer) if active_learning == False else TeacherDataset(ds['train'], tokenizer, device, T=T)
    print('Created `train_ds` with %d examples!'%len(train_ds))

    print('Preprocessing the Test Data...')
    test_ds = tokenize(ds['test'], tokenizer)
    print('Created `test_ds` with %d examples!'%len(test_ds))

    # Create the Collator
    data_collator =  DataCollatorWithPadding(tokenizer=tokenizer)

    return train_ds, test_ds, tokenizer, data_collator

def load_model(model_name, tokenizer, teacher=None, device='cuda:0'):
    """
        Loads the given HF model.
    """
    # Get the model's configuration.
    print('Loading configuraiton...')
    model_config = AutoConfig.from_pretrained(model_name, num_labels=2)

    # Get the model.
    print('Loading model...')
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

    if "gpt2" in model_name:
        # resize model embedding to match new tokenizer
        model.resize_token_embeddings(len(tokenizer))
        # fix model padding token id
        model.config.pad_token_id = model.config.eos_token_id

    if not teacher is None:
        model = ActiveLearningModel(model, tok=tokenizer, teacher=teacher)

    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`'%device)

    return model

# ----------------------------- Experiment Class ----------------------------- #
class Experiment():
    def __init__(self,
              N: int=200_000,
              test_ratio: float=.3,
              epochs: int=1,
              bs: int=64,
              optimizer: str='RAdam',
              lr: float=2e-5,
              wd: float=1e-3,
              warm_pct: float=0.0,
              active_learning: bool=False,
              T: int=10_000,
              aware_sampling: bool=False,
              aware_sampling_type: str='unif_140',
              SAVE_DIR: str="Global_Work/",
              BASE_MODEL: str="vinai/bertweet-base",
              DATA_PATH: str="data/twitter-datasets/",
              seed: int=42,
              device: str="cuda:0") -> None:
        # Data Args
        self.N = N
        self.test_ratio = test_ratio

        # Training Args
        self.epochs = epochs
        self.optimizer = optimizer
        self.bs = bs
        self.lr = lr
        self.wd = wd
        self.warm_pct = warm_pct

        # Active Learnings Args
        self.active_learning = active_learning
        self.T = T
        self.aware_sampling = aware_sampling
        self.aware_sampling_type = aware_sampling_type

        # Global Args
        self.SAVE_DIR = SAVE_DIR
        self.BASE_MODEL = BASE_MODEL
        self.DATA_PATH = DATA_PATH
        self.device = device

        # Set the experiment seed
        random.seed(seed)

        # Print the summary of the experiment
        self.summarize()

    def summarize(self):
        """
            Summarizes the experiment
        """
        print('Experiment summary:')
        print(f'- Base Model: {self.BASE_MODEL}')
        print('-'*30)
        print(f'- Train Set Size: {int(self.N*(1-self.test_ratio))} samples')
        print(f'- Test Set Size: {int(self.N*self.test_ratio)} samples')
        print('-'*30)
        as_txt = f'\U00002705 -> {self.aware_sampling_type}' if self.aware_sampling else '\U0000274C'
        print(f'- Aware Sampling: {as_txt}')
        al_txt = '\U00002705' if self.active_learning else '\U0000274C'
        print(f'- Active Learning: {al_txt}')
        print('-'*30)
        print(f'- Optimizer: {self.optimizer}')
        print(f'- Start Learning Rate: {self.lr}')
        warm_txt = f'\U00002705 -> {self.warm_pct:.0%} of the epochs' if (self.optimizer.lower() == 'radam') or (self.warm_pct > 0.0) else '\U0000274C'
        print(f'- Warmup: {warm_txt}')
   
    def finetune(self):
        """
            Fine-tune the experiment model.
        """
        # Load the data
        print(f"{'-'*30} Preparing the data {'-'*30}")
        train_ds, test_ds, tokenizer, data_collator = load_data(self.DATA_PATH, self.N, self.BASE_MODEL, self.test_ratio, \
                                                                active_learning=self.active_learning, aware_sampling=self.aware_sampling, aware_sampling_type=self.aware_sampling_type,\
                                                                    device=self.device)
        # Load the model
        print(f"{'-'*30} Preparing the model {'-'*30}")
        teacher = train_ds if self.active_learning else None
        model = load_model(self.BASE_MODEL, tokenizer, teacher=teacher, device=self.device)

        # Pushing the variables to the object
        self.model = model
        self.tokenizer = tokenizer

        # Number of steps
        training_steps = int(len(train_ds)*self.bs)
        warmup_steps = self.warm_pct * training_steps

        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd) if self.optimizer.lower() == 'adamw' else \
                        RAdam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps) if self.optimizer.lower() == 'adamw' else \
                            None
        

        print(f"{'-'*30} Training {'-'*30}")
        epochs_cnt = 1
        if self.active_learning:
            while(len(teacher) != 0):
                ## train for 1 epoch = 1 round of the DataLoader
                training_args = TrainingArguments(
                    output_dir=f"{self.SAVE_DIR}/{self.BASE_MODEL}",
                    per_device_train_batch_size=self.bs,
                    per_device_eval_batch_size=self.bs,
                    num_train_epochs=1,
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    remove_unused_columns=False
                )

                self.trainer = CustomTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=teacher,
                    eval_dataset=test_ds,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics
                )

                self.trainer.train()
                print("FINISHED EPOCH ==> updating")
                teacher.new_iter(add_randomness=True) ## teacher round
                model.model.save_pretrained(f"{self.SAVE_DIR}/{self.BASE_MODEL}/epoch_{epochs_cnt}/")
                epochs_cnt += 1
        else:
            # Training Arguments
            training_args = TrainingArguments(
                output_dir=f"{self.SAVE_DIR}/{self.BASE_MODEL}",
                per_device_train_batch_size=self.bs,
                per_device_eval_batch_size=self.bs,
                num_train_epochs=self.epochs,
                evaluation_strategy="epoch",
                save_strategy="epoch"
            )
            # Trainer
            self.trainer = CustomTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=test_ds,
                tokenizer=tokenizer,
                optimizers=(optimizer, lr_scheduler),
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )

            self.trainer.train()

        return model

    def predict(self, save=False):
        # Load the Tweet Test dataset
        with open(f"{self.DATA_PATH}/test_data.txt") as test_file:
            test_id, test_tweets = zip(*[(x.split(",")[0], ",".join(x.split(",")[1:])) for x in test_file.read().split("\n")])
        
        # Tokenization
        test_df = pd.DataFrame({'Id':test_id, 'tweet': test_tweets}).set_index("Id")
        test_ds = HFDataset.from_pandas(test_df)
        test_ds = tokenize(test_ds, self.tokenizer)

        # Prediction
        predictions = []
        Ids = []
        for i, test_sample in enumerate(test_ds):
            print(f"{i+1} / {len(test_ds)}", end="\r")
            input_ids = test_sample.get("input_ids")
            Ids.append(test_sample.get("Id"))
            attention_mask = test_sample.get("attention_mask")
            outputs = self.model(input_ids=torch.tensor(input_ids).squeeze(1), attention_mask=torch.tensor(attention_mask).squeeze(1))
            logits = outputs.get("logits").detach().cpu().numpy()
            predictions.append(logits)

        # Store
        predictions = [-1 if pred.argmax() == 0 else 1 for pred in predictions]
        pred_df = pd.DataFrame({"Id": Ids, "Prediction": predictions}).set_index("Id")
        
        # Save
        if save:
            pred_df.to_csv(f"{self.SAVE_DIR}/results/{self.model}.csv")

        return pred_df
       

    def run(self):
        """
            Run the whole experiment.
        """
        # Training
        self.finetune()

        # Predict
        self.predict(save=True)