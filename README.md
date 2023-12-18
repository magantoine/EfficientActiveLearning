# Harnessing the Potential of Pretrained Language Models and Active Learning for Tweets Sentiment Analysis

## **Main Results**
---


## **Using our codebase**
---
#### Available Pre-Trained Models
This codebase is built to be compatible with any HuggingFace listed model. You can look for available models on their page.

#### Experiments
- **Requirements**:
Here are the requirements to use our code:
```bash
pip install ...
```

- **Launch An Experiment**:
Our code is really simple to use. 

1) Specify your arguments in the **Parameters** section.
```python
# This launch an experiment using DistillBERT model with 10 000 samples using 3 epochs.
exp = Experiment(
    N=10_000,
    epochs=3,
    BASE_MODEL='distilbert-base-uncased'
)
```

2) Run the **Training** section to fine-tune your model.

3) Run the **Predict** section to predict your test data.




