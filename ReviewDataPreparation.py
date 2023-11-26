
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
import random
import numpy as np

class ReviewDataPreparation:
    def __init__(self, df, senti_df, model_name='bert-base-uncased', k=100):
        self.df = df
        self.senti_df = senti_df
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.k = k

    def encode_stars(self):
        self.df['stars'] = self.df['stars']
        one_hot_encoded = pd.get_dummies(self.df['stars'], prefix='star').astype(int)
        self.df = pd.concat([self.df, one_hot_encoded], axis=1)
        self.df = self.df.drop('stars', axis=1)

    def map_sentiments(self):
        sentiment_mapping = {'positive': 1, 'negative': 0}
        self.senti_df['sentiment'] = self.senti_df['sentiment'].map(sentiment_mapping)

    def merge_dataframes(self):
        self.df = pd.merge(self.df, self.senti_df, on='review_links', how='inner')
        self.df = self.df.dropna()

    def prepare_data(self, texts, numeric_features, labels):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        inputs_numeric = torch.tensor(numeric_features, dtype=torch.float32)
        labels = torch.tensor(labels)
        return TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs_numeric, labels)

    def get_texts(self):
        positive_texts = self.df[self.df['label'] == 1]['text'].tolist()
        unlabelled_texts = self.df[self.df['label'] == 0]['text'].tolist()
        spies_texts = random.sample(positive_texts, self.k)
        return positive_texts, unlabelled_texts, spies_texts

    def get_data(self):
        self.encode_stars()
        self.map_sentiments()
        self.merge_dataframes()

        numeric_features = self.df[['useful', 'funny', 'cool', 'star_1', 'star_2', 'star_3', 'star_4', 'star_5', 'sentiment']].values

        positive_texts, unlabelled_texts, spies_texts = self.get_texts()

        labels_spies = np.zeros((self.k, numeric_features.shape[1]))

        positive_data = self.prepare_data(positive_texts, numeric_features[self.df['label'] == 1], labels=[1] * len(positive_texts))
        mixed_spies_data = self.prepare_data(
            unlabelled_texts + spies_texts,
            np.vstack([numeric_features[self.df['label'] == 0], labels_spies]),
            labels=[0] * (len(unlabelled_texts) + self.k)
        )

        return positive_data, mixed_spies_data
    
