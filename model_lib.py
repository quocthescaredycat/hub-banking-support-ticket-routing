"""Pretrained classifier architectures for BANKING77."""

from torch import nn
from transformers import DistilBertModel, RobertaModel


class RoBERTa(nn.Module):
    """Fine-tune RoBERTa using its first token, <s>, for classification."""

    def __init__(self, num_labels=77, checkpoint="FacebookAI/roberta-base", dropout=0.1):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(checkpoint, add_pooling_layer=False)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(hidden[:, 0])


class DistilBERT(nn.Module):
    """Fine-tune DistilBERT using its [CLS] token for classification."""

    def __init__(self, num_labels=77, checkpoint="distilbert/distilbert-base-uncased", dropout=0.2):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(checkpoint)
        hidden_size = self.encoder.config.dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(hidden[:, 0])
