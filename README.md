# Emotion Detection AI

A Machine Learning and Natural Language Processing project that detects emotions from text.

## Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to classify text into six different emotions.

The model is trained using the DAIR.AI Emotion dataset containing 20,000 labeled text examples.

## Emotions

The model can detect:

- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

## Dataset

The dataset contains:

- 16,000 training examples
- 2,000 validation examples
- 2,000 testing examples

Dataset: DAIR.AI Emotion Dataset

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- Hugging Face Datasets

## Model

The project uses a machine learning pipeline:

Text → TF-IDF Vectorization → Logistic Regression → Emotion

## Performance

The trained model achieved:

**86.15% accuracy on the test dataset.**

## Project Structure

```text
emotion_detection_ai/
│
├── data/
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   └── emotions.csv
│
├── train.py
├── predict.py
├── download_dataset.py
├── emotion_model.pkl
├── requirements.txt
└── README.md