# Sentiment Classification Using Semi-Supervised Learning

This project presents a semi-supervised learning approach to sentiment classification using a small labeled dataset (from Kaggle) and a large unlabeled corpus (Amazon cellphone reviews). The pipeline incorporates active learning and contrastive learning to reduce manual labeling efforts while progressively improving model performance.

---

## Pipeline Overview

The end-to-end training pipeline is designed to iteratively improve model performance through uncertainty sampling and selective manual annotation.

<img width="1052" height="597" alt="image" src="https://github.com/user-attachments/assets/36d1186c-0936-4260-9e5b-574b45e385f7" />


### Steps:

1. **Labeled Dataset Initialization**  
   Begin with a small labeled dataset to train the initial model.

2. **Zero-Shot Inference**  
   Use the trained model to predict sentiments on an unlabeled Amazon review dataset.

3. **Uncertainty-Based Sampling**  
   Identify low-confidence predictions and select them for manual review.

4. **Manual Annotation**  
   Annotators label the selected samples, which are added to the labeled dataset.

5. **Iterative Fine-Tuning**  
   Retrain the model with the expanded dataset. First fine-tune the classifier, then the full encoder + classifier.

6. **Repeat**  
   The above cycle is repeated until the model reaches satisfactory performance.

---

## Model Architecture

The sentiment classification model consists of three main components: an encoder, a triplet loss head, and a classifier head.

<img width="761" height="474" alt="image" src="https://github.com/user-attachments/assets/cbd3db55-d059-4b19-bf00-1740e646cc01" />


### 1. Encoder

- **BERT (Frozen)**  
  Pre-trained BERT is used to generate contextual token embeddings. The weights remain frozen during training.

- **2-Layer BiLSTM**  
  Captures sequential information from token embeddings.

- **Attention Layer**  
  Assigns weights to important tokens for sentence-level representation.

- **Linear Projection**  
  Reduces the attention output to a fixed-size sentence embedding.

### 2. Triplet Loss Head

- **Triplet Loss (Batch Hard)**  
  Encourages semantically similar sentences to be close in embedding space, and dissimilar ones to be distant. Improves generalization and embedding quality.

### 3. Classifier Head

- **Three Fully Connected Layers**  
  Includes ReLU activation, Dropout, and Batch Normalization.

- **Output Layer**  
  Predicts one of three sentiment classes: Positive, Negative, or Neutral.

---

## Key Features

- Semi-supervised pipeline with minimal manual annotation.
- Metric learning via triplet loss to enhance representation quality.
- Modular training loop with uncertainty-based data selection.


