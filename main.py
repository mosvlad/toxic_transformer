import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
import warnings

warnings.filterwarnings('ignore')

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define constants
MODEL_NAME = "DeepPavlov/rubert-base-cased"  # Russian BERT model
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./russian_comment_classifier"
NUM_LABELS = 4  # NORMAL, INSULT, THREAT, OBSCENITY
LABELS = ["NORMAL", "INSULT", "THREAT", "OBSCENITY"]

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }


def parse_dataset(file_path):
    """
    Parse the dataset file into texts and labels
    """
    texts = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            label_part = line.split()[0]
            text = line[len(label_part) + 1:].strip()
            label_parts = label_part.split(",")

            # Create multi-hot encoding for labels
            label = [
                1 if "__label__NORMAL" in label_parts or "labelNORMAL" in label_parts else 0,
                1 if "__label__INSULT" in label_parts or "labelINSULT" in label_parts else 0,
                1 if "__label__THREAT" in label_parts or "labelTHREAT" in label_parts else 0,
                1 if "__label__OBSCENITY" in label_parts or "labelOBSCENITY" in label_parts else 0
            ]

            texts.append(text)
            labels.append(label)

    return texts, labels


def compute_metrics(p: EvalPrediction):
    """
    Compute metrics for multi-label classification
    """
    preds = (p.predictions > 0).astype(np.int32)
    labels = p.label_ids.astype(np.int32)

    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    # Per-class F1 scores
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    per_class_f1_dict = {f"f1_{LABELS[i]}": per_class_f1[i] for i in range(NUM_LABELS)}

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        **per_class_f1_dict
    }


def main():
    # Load and tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Parse dataset
    print("Parsing dataset...")
    file_path = "./dataset.txt"  # Update with your actual path
    texts, labels = parse_dataset(file_path)

    # Convert to pandas for easier manipulation
    df = pd.DataFrame({
        'text': texts,
        'labels': labels
    })

    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Training on {len(train_df)} examples, validating on {len(val_df)} examples")

    # Create datasets
    train_dataset = CommentDataset(
        train_df['text'].values,
        train_df['labels'].values.tolist(),
        tokenizer,
        MAX_LENGTH
    )

    val_dataset = CommentDataset(
        val_df['text'].values,
        val_df['labels'].values.tolist(),
        tokenizer,
        MAX_LENGTH
    )

    # Load pre-trained model
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Training the model...")
    trainer.train()

    # Evaluate the model
    print("Evaluating the model...")
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")

    # Save the model
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete!")


def predict(text, model_path=OUTPUT_DIR):
    """
    Predict the label for a single text
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Tokenize input
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.sigmoid(logits)

    # Convert to labels
    predicted_labels = (predictions > 0.5).int().numpy()[0]

    # Print results
    result_labels = []
    for i, label in enumerate(LABELS):
        if predicted_labels[i] == 1:
            result_labels.append(label)

    if not result_labels:
        # If no label exceeds threshold, pick the highest one
        max_idx = predictions[0].argmax().item()
        result_labels = [LABELS[max_idx]]

    return {
        'text': text,
        'predicted_labels': result_labels,
        'probabilities': {label: float(predictions[0][i]) for i, label in enumerate(LABELS)}
    }


# Example usage
if __name__ == "__main__":
    # Train the model
    main()

    # Example inference
    sample_text = "классный летунчик!"
    result = predict(sample_text)
    print(f"Sample text: {result['text']}")
    print(f"Predicted labels: {result['predicted_labels']}")
    print(f"Probabilities: {result['probabilities']}")