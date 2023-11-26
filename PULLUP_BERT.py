
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertModel

class PULLUPModel(nn.Module):
    def __init__(self, bert_model, numeric_input_size, num_labels):
        super(PULLUPModel, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.1)
        self.numeric_layers = nn.Sequential(
            nn.Linear(numeric_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.combined_layer = nn.Linear(768 + 32, num_labels)  # Adjust input size based on BERT output size

    def forward(self, input_ids, attention_mask, numeric_features):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled_output = torch.mean(bert_outputs, dim=1)
        pooled_output = self.dropout(pooled_output)

        numeric_outputs = self.numeric_layers(numeric_features)
        combined_features = torch.cat([pooled_output, numeric_outputs], dim=1)

        logits = self.combined_layer(combined_features)

        return logits

class PULLUPLoss(nn.Module):
    def __init__(self, alpha):
        super(PULLUPLoss, self).__init__()
        self.alpha = alpha
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, target_likelihoods, target_labels, target_entropies):
        loss_likelihood = self.criterion(target_likelihoods, target_labels)
        loss_entropy = torch.mean(target_entropies)
        loss = loss_likelihood - self.alpha * loss_entropy
        return loss

class PULLUPTrainer:
    def __init__(self, model, optimizer, criterion, num_mc_samples_train):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_mc_samples_train = num_mc_samples_train

    def mc_passes(self, input_ids, attention_mask, numeric_features):
        all_likelihoods_batch = []
        all_entropies_batch = []

        for _ in range(self.num_mc_samples_train):
            logits = self.model(input_ids, attention_mask, numeric_features)
            probabilities = torch.softmax(logits, dim=1)
            all_likelihoods_batch.extend(probabilities[:, 1].detach().cpu().numpy())
            predictive_entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
            all_entropies_batch.extend(predictive_entropy.detach().cpu().numpy())

        return all_likelihoods_batch, all_entropies_batch

    def compute_target_entropies_tensor(self, target_entropies):
        return torch.tensor(np.mean(target_entropies, axis=0).astype(np.float32), dtype=torch.float32, requires_grad=True)

    def compute_target_labels_tensor(self, target_likelihoods):
        target_labels = (np.mean(target_likelihoods, axis=0) > 0.5).astype(int)
        return torch.tensor(target_labels, dtype=torch.float32)

    def train_epoch(self, train_loader, device):
        self.model.train()
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, numeric_features, labels = batch
            input_ids, attention_mask, numeric_features, labels = input_ids.to(device), attention_mask.to(device), numeric_features.to(device), labels.to(device)

            all_likelihoods_batch, all_entropies_batch = self.mc_passes(input_ids, attention_mask, numeric_features)

            target_likelihoods_tensor = torch.tensor(np.mean(all_likelihoods_batch, axis=0).astype(np.float32), dtype=torch.float32, requires_grad=True).to(device)
            target_entropies_tensor = self.compute_target_entropies_tensor(all_entropies_batch).to(device)
            target_labels_tensor = self.compute_target_labels_tensor(all_likelihoods_batch).to(device)

            loss = self.criterion(target_likelihoods_tensor, target_labels_tensor, target_entropies_tensor)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


class PULLUPEvaluator:
    
    def __init__(self, model, num_mc_samples_eval):
        self.model = model
        self.num_mc_samples_eval = num_mc_samples_eval

    def mc_passes_eval(self, data_loader, device):
        all_likelihoods_eval = []
        all_entropies_eval = []

        with torch.no_grad():
            for _ in range(self.num_mc_samples_eval):
                likelihoods_batch_eval = []
                entropies_batch_eval = []

                for batch_eval in data_loader:
                    input_ids, attention_mask, numeric_features, _ = batch_eval
                    input_ids, attention_mask, numeric_features = input_ids.to(device), attention_mask.to(device), numeric_features.to(device)
                    model_output_eval = self.model(input_ids, attention_mask, numeric_features)
                    probabilities_eval = torch.softmax(model_output_eval, dim=1)
                    likelihoods_batch_eval.extend(probabilities_eval.detach().cpu().numpy())
                    entropy_eval = -torch.sum(probabilities_eval * torch.log(probabilities_eval + 1e-10), dim=1)
                    entropies_batch_eval.extend(entropy_eval.detach().cpu().numpy())

                all_likelihoods_eval.append(likelihoods_batch_eval)
                all_entropies_eval.append(entropies_batch_eval)

        return all_likelihoods_eval, all_entropies_eval

    def compute_mean_values(self, all_values):
        return np.mean(all_values, axis=0)

    def evaluate(self, data_loader, device):
        all_likelihoods_eval, all_entropies_eval = self.mc_passes_eval(data_loader, device)
        mean_likelihoods_eval = self.compute_mean_values(all_likelihoods_eval)
        mean_entropies_eval = self.compute_mean_values(all_entropies_eval)

        return mean_likelihoods_eval, mean_entropies_eval     
