# PULLUP-Fake-Review-Detection

Fake and deceptive reviews pose a significant challenge for online microblogging platforms like Yelp and TripAdvisor. Despite substantial progress in fake review detection over the years, existing solutions often face limitations, such as effectiveness on specific datasets or the need for extensive text feature mining.None of these solutions are particularly fit for large scale deployment which is something required by sites like Yelp. 

In response to these challenges, we present a novel solution leveraging Positive Unlabeled Learning with Bayesian optimization and the SPIES method. This approach aims to enhance the scalability and generalizability of fake review detection. Additionally, we implement Focal Loss to address potential imbalances in the ratio of genuine to fake reviews. 

We use a part of the open yelp dataset which we have manually labelled as the dataset to train our models, further we test it with the 'restaurant-reviews-dataset'(restaurant reviews) and the 'deceptive opinion dataset'(Hotel reviews) to test both unseeen performance on in-domain reviews and cross-domain reviews.

This project is a part of the course ISE 540 at the University of Southern California. Please email sanathsr@usc.edu before utilizing any part of the code and for access to any of the datasets used.  Thank You for visitng our repository

## Table of Contents

- [PULLUP Model](#pullup-model)
- [Classifier Model](#classifier-model)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Training and Evaluation](#training-and-evaluation)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Author](#author)

## PULLUP Model

The PULLUP model is designed to perform a specific task related to upvoting posts. It utilizes a combination of a pre-trained BERT model and additional numeric features. The training process includes Monte Carlo sampling passes to enhance performance.

## Classifier Model

The Classifier model is a text classification model based on the BERT architecture. It is intended for general text classification tasks and can be trained on labeled datasets.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
