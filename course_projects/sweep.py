#!/usr/bin/env python
# coding: utf-8

"""
Hyperparameter sweep script for CommonsenseQA models
"""

import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import wandb
import nltk
import gensim
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from datasets import load_dataset
from tqdm import trange
from huggingface_hub import hf_hub_download
import torcheval.metrics as metrics

# Parse arguments
parser = argparse.ArgumentParser(description='Run W&B sweeps for CommonsenseQA models')
parser.add_argument('--model_type', type=str, choices=['embeddings', 'rnn', 'both'], 
                    default='both', help='Which model to run sweeps for')
parser.add_argument('--trials', type=int, default=20, 
                    help='Number of trials to run for each sweep')
parser.add_argument('--sweep_method', type=str, choices=['random', 'grid', 'bayes'], 
                    default='bayes', help='Sweep search method')
args = parser.parse_args()

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create checkpoint directories
embeddings_checkpoints_path = "./checkpoints/embeddings"
rnn_checkpoints_path = "./checkpoints/rnn"
os.makedirs(embeddings_checkpoints_path, exist_ok=True)
os.makedirs(rnn_checkpoints_path, exist_ok=True)

# Download NLTK data if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load fasttext embeddings
print("Loading fasttext embeddings...")
model_path = hf_hub_download("facebook/fasttext-en-vectors", "model.bin")
fasttext_model = gensim.models.fasttext.load_facebook_model(model_path)
wv = fasttext_model.wv
embedding_dim = wv.vector_size
print(f"Loaded embeddings with dimension {embedding_dim}")

# Load data
print("Loading datasets...")
train = load_dataset("tau/commonsense_qa", split="train[:-1000]")
valid = load_dataset("tau/commonsense_qa", split="train[-1000:]")
test = load_dataset("tau/commonsense_qa", split="validation")
print(f"Dataset sizes - Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

# Preprocessing functions
def preprocess_text(text):
    """Tokenize text into words"""
    return word_tokenize(text)

def get_averaged_sentence_embedding(sentence):
    """Convert sentence to averaged word embeddings"""
    tokens = preprocess_text(sentence)
    word_vectors = [wv[word] for word in tokens]
    return np.mean(word_vectors, axis=0)

def get_sentence_embedding(sentence):
    """Convert sentence to sequence of word embeddings"""
    tokens = preprocess_text(sentence)
    word_vectors = [wv[word] for word in tokens]
    return np.array(word_vectors)

def answer_key_to_index(answer_key):
    """Convert answer key (A-E) to index (0-4)"""
    return ord(answer_key) - ord("A")

# Define special tokens for RNN model
special_tokens = {
    "[EOQ]": np.random.uniform(-0.1, 0.1, embedding_dim),
    "[EOC1]": np.random.uniform(-0.1, 0.1, embedding_dim),
    "[EOC2]": np.random.uniform(-0.1, 0.1, embedding_dim),
    "[EOC3]": np.random.uniform(-0.1, 0.1, embedding_dim),
    "[EOC4]": np.random.uniform(-0.1, 0.1, embedding_dim),
}

# Add special tokens to word embeddings if not already there
for token, vec in special_tokens.items():
    if token not in wv.key_to_index:
        wv.add_vectors([token], [vec])

# Dataset classes
class EmbeddingsCommonsenseQADataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        average_question_embedding = get_averaged_sentence_embedding(example["question"])
        average_choice_embeddings = [get_averaged_sentence_embedding(choice) for choice in example["choices"]["text"]]
        average_choice_embeddings = np.array(average_choice_embeddings)

        question_tensor = torch.tensor(average_question_embedding).float()
        choices_tensor = torch.tensor(average_choice_embeddings).float()
        answer_index = answer_key_to_index(example["answerKey"])
        return question_tensor, choices_tensor, torch.tensor(answer_index).long()

class RNNCommonsenseQADataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
        self.special_separator_tokens = {
            "EOQ": torch.tensor(wv["[EOQ]"]).unsqueeze(0),
            "EOC1": torch.tensor(wv["[EOC1]"]).unsqueeze(0),
            "EOC2": torch.tensor(wv["[EOC2]"]).unsqueeze(0),
            "EOC3": torch.tensor(wv["[EOC3]"]).unsqueeze(0),
            "EOC4": torch.tensor(wv["[EOC4]"]).unsqueeze(0),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        question_embedding = get_sentence_embedding(example["question"])
        question_embedding = torch.tensor(question_embedding)
        choice_embeddings = [get_sentence_embedding(choice) for choice in example["choices"]["text"]]
        choice_embeddings = [torch.tensor(choice_embedding) for choice_embedding in choice_embeddings]

        concatenated = torch.cat([
            question_embedding,
            self.special_separator_tokens["EOQ"],
            choice_embeddings[0],
            self.special_separator_tokens["EOC1"],
            choice_embeddings[1],
            self.special_separator_tokens["EOC2"],
            choice_embeddings[2],
            self.special_separator_tokens["EOC3"],
            choice_embeddings[3],
            self.special_separator_tokens["EOC4"],
            choice_embeddings[4],
        ])

        answer_index = answer_key_to_index(example["answerKey"])
        return concatenated, torch.tensor(answer_index).long()

# Collate function for RNN model
def pad_collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Get sequence lengths before padding
    lengths = torch.tensor([len(seq) for seq in sequences])

    # Pad sequences to max length in current batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    # Return padded sequences, original lengths, and labels
    return padded_sequences, lengths, torch.tensor(labels)

# Model definitions
class WordEmbeddingQAClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_prob):
        super(WordEmbeddingQAClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(dropout_prob)

        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, question, choices):
        # question: (batch_size, embedding_dim)
        # choices: (batch_size, 5, embedding_dim)

        # expand question to match the choices dimension
        question_expanded = question.unsqueeze(1).expand(-1, choices.size(1), -1)

        # concatenate question and choice embeddings
        combined = torch.cat((question_expanded, choices), dim=2)

        # pass through the classifier
        x = self.fc1(combined)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x.squeeze(-1)  # (batch_size, 5)

class RNNQAClassifier(nn.Module):
    def __init__(self, embedding_dim, num_layers, bidirectional, rnn_hidden_dim, classifier_hidden_dim, dropout_prob):
        super(RNNQAClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.classifier_hidden_dim = classifier_hidden_dim
        self.num_layers = num_layers

        self.num_directions = 2 if bidirectional else 1

        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_directions * rnn_hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(classifier_hidden_dim, 5)
        )

    def forward(self, padded_sequences, sequence_lengths):
        # Pack the sequences based on their actual lengths
        packed_sequences = pack_padded_sequence(
            padded_sequences, 
            sequence_lengths.cpu(), 
            batch_first=True,
            enforce_sorted=False
        )

        # Process with LSTM
        _, (h_n, _) = self.rnn(packed_sequences)

        # If bidirectional, concatenate the final states from both directions
        if self.num_directions == 2:
            hidden_final = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            hidden_final = h_n[-1,:,:]

        # Pass through classifier
        out = self.classifier(hidden_final)
        return out

# Utility functions
def save_checkpoint(model, optimizer, epoch, scheduler, save_model_path, checkpoint_name):
    checkpoint_name = f"{checkpoint_name}.pt"
    save_path = os.path.join(save_model_path, checkpoint_name)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }, save_path)

def evaluate_model(model, test_loader, model_type="embeddings"):
    """Evaluate model performance on a dataset"""
    model.eval()

    # Initialize metrics
    accuracy_metric = metrics.MulticlassAccuracy(num_classes=5)
    accuracy_metric.to(device)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = [tensor.to(device) for tensor in batch]
            y_batch = batch.pop(-1)  # Get labels

            # Forward pass
            outputs = model(*batch)
            
            # Update accuracy metric
            accuracy_metric.update(outputs, y_batch)
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    # Compute accuracy
    accuracy = accuracy_metric.compute().item()
    
    return {
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels
    }

# Sweep configurations
embeddings_sweep_config = {
    'method': args.sweep_method,
    'name': f'embeddings-sweep-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'hidden_dim': {
            'values': [32, 64, 128, 256]
        },
        'dropout_prob': {
            'min': 0.1,
            'max': 0.5
        },
        'initial_learning_rate': {
            'min': 1e-5,
            'max': 1e-3,
            'distribution': 'log_uniform_values'
        },
        'max_learning_rate': {
            'min': 5e-5,
            'max': 5e-3,
            'distribution': 'log_uniform_values'
        },
        'weight_decay': {
            'values': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        }
    }
}

rnn_sweep_config = {
    'method': args.sweep_method,
    'name': f'rnn-sweep-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'rnn_hidden_dim': {
            'values': [64, 128, 256]
        },
        'classifier_hidden_dim': {
            'values': [64, 128, 256]
        },
        'bidirectional': {
            'values': [True, False]
        },
        'dropout_prob': {
            'min': 0.1,
            'max': 0.5
        },
        'initial_learning_rate': {
            'min': 1e-5,
            'max': 1e-3,
            'distribution': 'log_uniform_values'
        },
        'max_learning_rate': {
            'min': 5e-5,
            'max': 5e-3,
            'distribution': 'log_uniform_values'
        },
        'weight_decay': {
            'values': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        }
    }
}

# Sweep training functions
def train_embeddings_model():
    """Training function for embeddings model sweep"""
    run_name = f"word-embedding-qa-{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
    with wandb.init() as run:
        run.name = f"sweep-{run_name}"
        config = wandb.config
        
        # Create datasets and dataloaders
        embeddings_train_dataset = EmbeddingsCommonsenseQADataset(train)
        embeddings_valid_dataset = EmbeddingsCommonsenseQADataset(valid)
        
        batch_size = 2048
        embeddings_train_loader = DataLoader(
            embeddings_train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=8
        )
        embeddings_valid_loader = DataLoader(
            embeddings_valid_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=8
        )
        
        # Create model
        model = WordEmbeddingQAClassifier(
            embedding_dim=embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout_prob=config.dropout_prob
        )
        model = model.to(device)
        
        # Watch the model
        wandb.watch(model, log="all")
        
        # Set up optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.initial_learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.max_learning_rate,
            steps_per_epoch=len(embeddings_train_loader),
            epochs=50
        )
        
        # Training loop
        best_val_accuracy = 0.0
        epochs = 50
        
        for epoch in (pbar := trange(epochs)):
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            
            # Initialize metrics for this epoch
            train_accuracy_metric = metrics.MulticlassAccuracy(num_classes=5)
            val_accuracy_metric = metrics.MulticlassAccuracy(num_classes=5)
            train_accuracy_metric.to(device)
            val_accuracy_metric.to(device)
            
            # Training phase
            model.train()
            train_total_loss = 0.0
            
            for batch in embeddings_train_loader:
                optimizer.zero_grad()
                
                # Move batch to device
                question_tensor, choices_tensor, answer_index = [t.to(device) for t in batch]
                
                # Forward pass
                outputs = model(question_tensor, choices_tensor)
                loss = criterion(outputs, answer_index)
                train_total_loss += loss.item()
                
                # Update metrics
                train_accuracy_metric.update(outputs, answer_index)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            # Calculate training statistics
            avg_train_loss = train_total_loss / len(embeddings_train_loader)
            train_accuracy = train_accuracy_metric.compute().item()
            
            # Validation phase
            model.eval()
            val_total_loss = 0.0
            
            with torch.no_grad():
                for batch in embeddings_valid_loader:
                    # Move batch to device
                    question_tensor, choices_tensor, answer_index = [t.to(device) for t in batch]
                    
                    # Forward pass
                    outputs = model(question_tensor, choices_tensor)
                    loss = criterion(outputs, answer_index)
                    val_total_loss += loss.item()
                    
                    # Update metrics
                    val_accuracy_metric.update(outputs, answer_index)
            
            # Calculate validation statistics
            avg_val_loss = val_total_loss / len(embeddings_valid_loader)
            val_accuracy = val_accuracy_metric.compute().item()
            
            # Save checkpoints
            checkpoint_base = f"embeddings_sweep_{run.id}_epoch{epoch}"
            # save_checkpoint(model, optimizer, epoch, scheduler, embeddings_checkpoints_path, checkpoint_base)
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_checkpoint(model, optimizer, epoch, scheduler, embeddings_checkpoints_path, f"best_{run.id}")
            
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            pbar.set_postfix({
                "train_loss": f"{avg_train_loss:.4f}", 
                "train_acc": f"{train_accuracy:.4f}", 
                "val_acc": f"{val_accuracy:.4f}"
            })
            
            # Early stopping
            if epoch > 20 and val_accuracy < 0.2:
                print(f"Early stopping run {run.id} - validation accuracy too low")
                break
        
        # Log final best validation accuracy
        wandb.log({"best_val_accuracy": best_val_accuracy})

def train_rnn_model():
    """Training function for RNN model sweep"""
    run_name = f"rnn-qa-{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
    with wandb.init() as run:
        config = wandb.config
        run.name = f"sweep-{run_name}"
        
        # Create datasets and dataloaders
        rnn_train_dataset = RNNCommonsenseQADataset(train)
        rnn_valid_dataset = RNNCommonsenseQADataset(valid)
        
        batch_size = 512
        rnn_train_loader = DataLoader(
            rnn_train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=8, 
            collate_fn=pad_collate_fn
        )
        rnn_valid_loader = DataLoader(
            rnn_valid_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=8, 
            collate_fn=pad_collate_fn
        )
        
        # Create model
        model = RNNQAClassifier(
            embedding_dim=embedding_dim,
            num_layers=2,
            bidirectional=config.bidirectional,
            rnn_hidden_dim=config.rnn_hidden_dim,
            classifier_hidden_dim=config.classifier_hidden_dim,
            dropout_prob=config.dropout_prob
        )
        model = model.to(device)
        
        # Watch the model
        wandb.watch(model, log="all")
        
        # Set up optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.initial_learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.max_learning_rate,
            steps_per_epoch=len(rnn_train_loader),
            epochs=50
        )
        
        # Training loop
        best_val_accuracy = 0.0
        epochs = 50
        
        for epoch in (pbar := trange(epochs)):
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            
            # Initialize metrics for this epoch
            train_accuracy_metric = metrics.MulticlassAccuracy(num_classes=5)
            val_accuracy_metric = metrics.MulticlassAccuracy(num_classes=5)
            train_accuracy_metric.to(device)
            val_accuracy_metric.to(device)
            
            # Training phase
            model.train()
            train_total_loss = 0.0
            
            for batch in rnn_train_loader:
                optimizer.zero_grad()
                
                # Move batch to device
                padded_sequences, lengths, labels = [t.to(device) for t in batch]
                
                # Forward pass
                outputs = model(padded_sequences, lengths)
                loss = criterion(outputs, labels)
                train_total_loss += loss.item()
                
                # Update metrics
                train_accuracy_metric.update(outputs, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            # Calculate training statistics
            avg_train_loss = train_total_loss / len(rnn_train_loader)
            train_accuracy = train_accuracy_metric.compute().item()
            
            # Validation phase
            model.eval()
            val_total_loss = 0.0
            
            with torch.no_grad():
                for batch in rnn_valid_loader:
                    # Move batch to device
                    padded_sequences, lengths, labels = [t.to(device) for t in batch]
                    
                    # Forward pass
                    outputs = model(padded_sequences, lengths)
                    loss = criterion(outputs, labels)
                    val_total_loss += loss.item()
                    
                    # Update metrics
                    val_accuracy_metric.update(outputs, labels)
            
            # Calculate validation statistics
            avg_val_loss = val_total_loss / len(rnn_valid_loader)
            val_accuracy = val_accuracy_metric.compute().item()
            
            # Save checkpoints
            checkpoint_base = f"rnn_sweep_{run.id}_epoch{epoch}"
            # save_checkpoint(model, optimizer, epoch, scheduler, rnn_checkpoints_path, checkpoint_base)
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_checkpoint(model, optimizer, epoch, scheduler, rnn_checkpoints_path, f"best_{run.id}")
            
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            pbar.set_postfix({
                "train_loss": f"{avg_train_loss:.4f}", 
                "train_acc": f"{train_accuracy:.4f}", 
                "val_acc": f"{val_accuracy:.4f}"
            })
            
            # Early stopping
            if epoch > 20 and val_accuracy < 0.19:
                print(f"Early stopping run {run.id} - validation accuracy too low")
                break
        
        # Log final best validation accuracy
        wandb.log({"best_val_accuracy": best_val_accuracy})

# Main execution
if __name__ == "__main__":
    # Login to wandb
    wandb.login()
    
    try:
        # Run embeddings model sweep
        if args.model_type in ['embeddings', 'both']:
            print("\n=== Starting Embeddings Model Sweep ===")
            embeddings_sweep_id = wandb.sweep(
                embeddings_sweep_config,
                project="hslu-fs25-nlp-qa",
                entity="dhodel-hslu-nlp"
            )
            print(f"Embeddings sweep created with ID: {embeddings_sweep_id}")
            wandb.agent(embeddings_sweep_id, train_embeddings_model, count=args.trials)
            print(f"Embeddings sweep completed with {args.trials} trials")
        
        # Run RNN model sweep
        if args.model_type in ['rnn', 'both']:
            print("\n=== Starting RNN Model Sweep ===")
            rnn_sweep_id = wandb.sweep(
                rnn_sweep_config,
                project="hslu-fs25-nlp-qa",
                entity="dhodel-hslu-nlp"
            )
            print(f"RNN sweep created with ID: {rnn_sweep_id}")
            wandb.agent(rnn_sweep_id, train_rnn_model, count=args.trials)
            print(f"RNN sweep completed with {args.trials} trials")
        
        print("\n=== All sweeps completed successfully ===")
        
    except Exception as e:
        print(f"Error during sweep: {e}")
        raise e