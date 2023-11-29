# PULLUP-Fake-Review-Detection

Fake and deceptive reviews pose a significant challenge for online microblogging platforms like Yelp and TripAdvisor. Despite substantial progress in fake review detection over the years, existing solutions often face limitations, such as effectiveness on specific datasets or the need for extensive text feature mining.None of these solutions are particularly fit for large scale deployment which is something required by sites like Yelp. 

In response to these challenges, we present a novel solution leveraging Positive Unlabeled Learning with Bayesian optimization and the SPIES method. This approach aims to enhance the scalability and generalizability of fake review detection. Additionally, we implement Focal Loss to address potential imbalances in the ratio of genuine to fake reviews. 

We use a part of the open yelp dataset which we have manually labelled as the dataset to train our models, further we test it with the 'restaurant-reviews-dataset'(restaurant reviews) and the 'deceptive opinion dataset'(Hotel reviews) to test both unseeen performance on in-domain reviews and cross-domain reviews.

This project is a part of the course ISE 540 at the University of Southern California. Please email sanathsr@usc.edu before utilizing any part of the code and for access to any of the datasets used.  Thank You for visitng our repository

## Table of Contents
- [PU Learning and Spies Algorithm](#pu-learning-and-spies-algoirthm)
- [PULLUP Model](#pullup-model)
- [Classifier Model](#classifier-model)
- [Installation](#installation)
- [Usage](#usage)
- [Training and Evaluation](#training-and-evaluation)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Author](#author)

## Bayesian PU Learning and Spies Algorithm
![Spies](spies.png)
We introduce an uncertainty-powered PU learning approach using the SPIES method. Our workflow involves training an initial model on a Positive Set (P) and a Mix and Spies set (MS). One of the major disadvantages of the SPIES method is that it requires a large number of labeled spies for the algorithm to be effective. We work around this by using Monte Carlo dropout, which induces multiple forward passes of each sample, increasing the variance and uncertainty in predictions.

We utilize the SPIES method to separate likely negative (N) from likely positive (Q) instances and train a second classifier using N as the negative class and Positive (P + Q) as the positive class. Applied to fake review detection in text data, the methodology has proven effective in identifying deceptive content. Explore the code for customization and adaptation to your use case.

## PULLUP Model

The PULLUP model is designed to perform a specific task related to upvoting posts. It utilizes a combination of a pre-trained BERT model and additional numeric features. The training process includes Monte Carlo sampling passes to enhance performance.

## Classifier Model

The Classifier model is a text classification model based on the BERT architecture. It is intended for general text classification tasks and can be trained on labeled datasets.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
