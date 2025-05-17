# 📊 Thesis Sentiment Analysis: Greek Twitter Insights

Welcome to the **Thesis Sentiment Analysis** project! This repository contains tools and models for analyzing sentiment in Greek Twitter data using advanced techniques. Our main focus is on utilizing BERT, a powerful deep learning model, alongside traditional machine learning methods to classify sentiments effectively.

[![Download Releases](https://img.shields.io/badge/Download%20Releases-Here-blue)](https://github.com/ceyhun262613/thesis-sentiment-analysis/releases)

## 📚 Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Models](#models)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

In the age of social media, understanding public sentiment is crucial. This project aims to classify sentiments expressed in Greek tweets. We leverage the BERT model, which has shown remarkable performance in various natural language processing tasks, to analyze opinions and emotions in tweets.

## Technologies Used

This project employs the following technologies:

- **BERT**: A transformer model that provides state-of-the-art results in NLP tasks.
- **Logistic Regression**: A traditional machine learning model for binary classification.
- **Naive Bayes**: Another machine learning model useful for text classification.
- **Hugging Face Transformers**: A library that simplifies working with transformer models.
- **Python 3**: The programming language used for implementation.
- **Cross-Validation**: A technique to evaluate model performance.
- **Deep Learning**: For fine-tuning the BERT model.

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ceyhun262613/thesis-sentiment-analysis.git
   cd thesis-sentiment-analysis
   ```

2. **Install Dependencies**:
   Make sure you have Python 3 installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   You can find the dataset in the "data" folder. If you need the latest version, visit the [Releases section](https://github.com/ceyhun262613/thesis-sentiment-analysis/releases) to download it.

## How It Works

The sentiment analysis process involves several steps:

1. **Data Collection**: We collect tweets from the Twitter API that contain specific keywords.
2. **Preprocessing**: The tweets are cleaned and preprocessed to remove noise and irrelevant information.
3. **Model Training**: We train the BERT model and traditional classifiers using the preprocessed data.
4. **Sentiment Classification**: The trained models classify the sentiments as positive, negative, or neutral.
5. **Evaluation**: We evaluate the models using metrics like accuracy, precision, recall, and F1 score.

## Models

### BERT Model

The BERT model used in this project is a Greek uncased version. It is fine-tuned on our dataset to improve sentiment classification performance.

### Traditional Machine Learning Models

1. **Logistic Regression**: This model provides a simple yet effective way to classify sentiments based on linear decision boundaries.
2. **Naive Bayes**: A probabilistic model that assumes independence among features, making it efficient for text classification.

## Evaluation

We assess model performance using cross-validation. This technique helps ensure that our models generalize well to unseen data. We report the following metrics:

- **Accuracy**: The proportion of correct predictions.
- **Precision**: The ratio of true positive predictions to the total predicted positives.
- **Recall**: The ratio of true positive predictions to the total actual positives.
- **F1 Score**: The harmonic mean of precision and recall.

## Usage

To use the trained models for sentiment analysis, run the following command:

```bash
python predict.py --input "Your tweet here"
```

Replace `"Your tweet here"` with the tweet you want to analyze. The model will output the predicted sentiment.

## Contributing

We welcome contributions to this project. If you want to help, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch to your fork.
5. Open a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or suggestions, please reach out:

- **Author**: Ceyhun
- **Email**: ceyhun@example.com

Thank you for your interest in the Thesis Sentiment Analysis project! Explore the [Releases section](https://github.com/ceyhun262613/thesis-sentiment-analysis/releases) for the latest updates and downloads.