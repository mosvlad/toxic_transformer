import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define constants
MODEL_PATH = "./model"  # Path to your saved model
MAX_LENGTH = 128
LABELS = ["NORMAL", "INSULT", "THREAT", "OBSCENITY"]


def classify_comment(text, model_path=MODEL_PATH, threshold=0.5):
    """
    Classify a single Russian comment

    Args:
        text (str): The comment text to classify
        model_path (str): Path to the saved model
        threshold (float): Probability threshold for positive classification

    Returns:
        dict: Dictionary containing predictions and probabilities
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
        probabilities = torch.sigmoid(logits)

    # Convert to labels
    predicted_labels = (probabilities > threshold).int().numpy()[0]

    # Get results
    result_labels = []
    for i, label in enumerate(LABELS):
        if predicted_labels[i] == 1:
            result_labels.append(label)

    if not result_labels:
        # If no label exceeds threshold, pick the highest one
        max_idx = probabilities[0].argmax().item()
        result_labels = [LABELS[max_idx]]

    return {
        'text': text,
        'predicted_labels': result_labels,
        'probabilities': {label: float(probabilities[0][i]) for i, label in enumerate(LABELS)}
    }


def classify_batch(texts, model_path=MODEL_PATH, threshold=0.5, batch_size=16):
    """
    Classify a batch of Russian comments

    Args:
        texts (list): List of comment texts to classify
        model_path (str): Path to the saved model
        threshold (float): Probability threshold for positive classification
        batch_size (int): Batch size for processing

    Returns:
        list: List of dictionaries containing predictions and probabilities
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    results = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Make predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)

        # Process each result in the batch
        for j, text in enumerate(batch_texts):
            probs = probabilities[j]
            predicted_labels = (probs > threshold).int().numpy()

            # Get results
            result_labels = []
            for k, label in enumerate(LABELS):
                if predicted_labels[k] == 1:
                    result_labels.append(label)

            if not result_labels:
                # If no label exceeds threshold, pick the highest one
                max_idx = probs.argmax().item()
                result_labels = [LABELS[max_idx]]

            results.append({
                'text': text,
                'predicted_labels': result_labels,
                'probabilities': {label: float(probs[k]) for k, label in enumerate(LABELS)}
            })

    return results


# Example usage
if __name__ == "__main__":
    # Example of classifying a single comment
    test_comment = "беги по радуге бедняга.сколько мучился..."
    result = classify_comment(test_comment)
    print(f"Single comment classification:")
    print(f"Text: {result['text']}")
    print(f"Predicted labels: {', '.join(result['predicted_labels'])}")
    print(f"Probabilities: {result['probabilities']}")
    print()

    # Example of batch classification
    test_batch = [
        "Вот сволочи,так издеваться над животным,где жалость.И такие скоты живут рядом."]
    batch_results = classify_batch(test_batch)
    print(f"Batch classification results:")
    for i, res in enumerate(batch_results):
        print(f"Text {i + 1}: {res['text']}")
        print(f"Predicted labels: {', '.join(res['predicted_labels'])}")
        print(f"Probabilities: {res['probabilities']}")
        print()