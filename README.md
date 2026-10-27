# AI Internship Projects (Codec Technologies)

## 1. Handwritten Digit Recognizer
A Convolutional Neural Network (CNN) trained on the MNIST dataset to recognise
handwritten digits (0-9).
- Layers: 2x Conv2D + MaxPooling, Flatten, Dropout, Dense (softmax)
- Test accuracy achieved: **98.84%**
- Files: `digit_recognizer.py`, `predictions.png`

## 2. Spam Email Classifier
Detects spam messages from their text using TF-IDF features and a
Multinomial Naive Bayes classifier, trained on the UCI SMS Spam Collection.
- Prints accuracy, classification report and confusion matrix
- Tests the model on new example messages
- File: `spam_classifier.py`

## Setup

pip install -r requirements.txt
