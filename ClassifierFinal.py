
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F
from transformers import AdamW, BertForSequenceClassification
from sklearn.metrics import f1_score, recall_score, precision_score

class ClassifierModel(nn.Module):
    def __init__(self, num_labels=2):
        super(ClassifierModel, self).__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)  # Removed .float()
        return logits

class ClassifierTrainer:
    def __init__(self, model, loss_fn, optimizer, dataloader, test_loader=None, device='cuda'):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.test_loader = test_loader
        self.device = device

    def train(self, num_epochs=1):
        for epoch in range(num_epochs):
            total_loss = 0.0

            self.model.train()
            for batch in self.dataloader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.squeeze(-1)
                labels = labels.float()
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}')
        
        return self.model
