
# Sentiment Analysis of Amazon Beauty Product Reviews

## Project Overview

This project implements and evaluates two Natural Language
Processing (NLP) architectures for three-class sentiment
classification of Amazon Beauty product reviews.

The sentiment classes are:
- Negative (1–2 stars)
- Neutral (3 stars)
- Positive (4–5 stars)

The project was developed for the IU course
Project: NLP and Computer Vision (DLMAIEAIS02).

## Dataset

Source: Amazon Reviews 2023 - All_Beauty

- Initial subset: 5,000 reviews
- Cleaned dataset: 4,947 reviews
- Training samples: 3,462
- Validation samples: 742
- Test samples: 743

The dataset was split using stratified sampling.

## Models Implemented

### 1. TF-IDF + Logistic Regression

A traditional machine learning baseline using:
- TF-IDF vectorization
- Unigrams and bigrams
- Class-balanced Logistic Regression

### 2. DistilBERT

A pretrained transformer fine-tuned for three-class
sentiment classification.

Configuration:
- Model: distilbert-base-uncased
- Epochs: 2
- Batch size: 8
- Learning rate: 2e-5
- Maximum sequence length: 128

## Final Test Results

| Metric | Logistic Regression | DistilBERT |
|---|---:|---:|
| Accuracy | 0.7483 | 0.8560 |
| Macro Precision | 0.5497 | 0.6676 |
| Macro Recall | 0.5917 | 0.5998 |
| Macro F1 | 0.5652 | 0.6156 |

DistilBERT achieved higher overall accuracy and macro F1.

However, Neutral sentiment remained challenging,
and Logistic Regression achieved better Neutral recall.

## Project Notebooks

1. `01_data_exploration.ipynb`
   - Data exploration
   - Cleaning and sentiment labeling
   - Train/validation/test splitting

2. `02_logistic_regression.ipynb`
   - TF-IDF vectorization
   - Logistic Regression training
   - Validation evaluation

3. `03_distilbert.ipynb`
   - Tokenization
   - DistilBERT fine-tuning
   - Training experiments
   - Validation and model selection

4. `04_final_model_comparison.ipynb`
   - Final test evaluation
   - Model comparison
   - Confusion matrices
   - Error analysis

## Installation

Create a Python virtual environment:

python -m venv .venv

Activate it on Windows PowerShell:

.venv\Scripts\Activate.ps1

Install dependencies:

python -m pip install -r requirements.txt

## Running the Project

Open the project in VS Code or Jupyter.

Select the Python environment and open the notebooks
in the `notebooks` folder.

The saved models can be loaded for evaluation and
prediction without repeating the training process.

## Limitations

- The dataset is imbalanced toward Positive reviews.
- Neutral sentiment is difficult to classify.
- Sentiment labels are derived from star ratings.
- Training was performed on a limited dataset subset.
- DistilBERT training is computationally expensive on CPU.

## Future Improvements

- Investigate improved Neutral-class classification.
- Evaluate additional data-balancing approaches.
- Explore larger and more diverse review datasets.
- Perform further error analysis.

## Frameworks

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Matplotlib
- Jupyter Notebook
