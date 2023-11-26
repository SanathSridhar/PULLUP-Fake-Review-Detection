
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
import torch

class ClassifierDataPreparation:
    def __init__(self, positive_texts, unlabelled_texts, spies_texts, pseudo_labels):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.pseudo_labelled_data = self.prepare_data_classifier(unlabelled_texts + spies_texts, pseudo_labels)
        self.positive_classifier_data = self.prepare_data_classifier(positive_texts, labels=[1] * len(positive_texts))
        self.classification_dataset = self.positive_classifier_data + self.pseudo_labelled_data
        self.classification_loader = DataLoader(self.classification_dataset, batch_size=1, shuffle=True)

    def prepare_data_classifier(self, texts, labels):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        labels = torch.tensor(labels)
        return TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
