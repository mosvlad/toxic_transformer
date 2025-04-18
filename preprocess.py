import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Download Russian stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Get Russian stopwords
russian_stopwords = stopwords.words('russian')


def clean_text(text):
    """
    Clean text by removing URLs, special characters, etc.
    """
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def parse_and_preprocess_dataset(file_path, clean=True, remove_stopwords=False):
    """
    Parse and preprocess the dataset

    Args:
        file_path (str): Path to the dataset file
        clean (bool): Whether to clean the text
        remove_stopwords (bool): Whether to remove stopwords

    Returns:
        pandas.DataFrame: Preprocessed dataframe
    """
    texts = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                label_part = line.split()[0]
                text = line[len(label_part) + 1:].strip()

                # Handle different label formats
                if label_part.startswith("__label__"):
                    label_parts = label_part.split(",")
                elif label_part.startswith("label"):
                    # Convert from labelNORMAL to __label__NORMAL format
                    label_parts = ["__" + label_part]
                else:
                    continue  # Skip invalid lines

                # Create multi-hot encoding for labels
                label = [
                    1 if "__label__NORMAL" in label_parts or "labelNORMAL" in label_parts else 0,
                    1 if "__label__INSULT" in label_parts or "labelINSULT" in label_parts else 0,
                    1 if "__label__THREAT" in label_parts or "labelTHREAT" in label_parts else 0,
                    1 if "__label__OBSCENITY" in label_parts or "labelOBSCENITY" in label_parts else 0
                ]

                # Preprocess text if enabled
                if clean:
                    text = clean_text(text)

                # Remove stopwords if enabled
                if remove_stopwords:
                    text = ' '.join([word for word in text.split() if word.lower() not in russian_stopwords])

                texts.append(text)
                labels.append(label)
            except Exception as e:
                print(f"Error processing line: {line}")
                print(f"Error: {e}")

    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'NORMAL': [label[0] for label in labels],
        'INSULT': [label[1] for label in labels],
        'THREAT': [label[2] for label in labels],
        'OBSCENITY': [label[3] for label in labels]
    })

    # Add a column with all labels in a list
    df['labels'] = labels

    return df


def analyze_dataset(df):
    """
    Analyze the dataset and show statistics
    """
    # Basic statistics
    print(f"Dataset size: {len(df)} comments")

    # Label distribution
    label_counts = {
        'NORMAL': df['NORMAL'].sum(),
        'INSULT': df['INSULT'].sum(),
        'THREAT': df['THREAT'].sum(),
        'OBSCENITY': df['OBSCENITY'].sum()
    }

    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"{label}: {count} ({count / len(df) * 100:.2f}%)")

    # Plot label distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()))
    plt.title('Label Distribution')
    plt.ylabel('Count')
    plt.xlabel('Label')
    plt.tight_layout()
    plt.savefig('label_distribution.png')

    # Text length statistics
    df['text_length'] = df['text'].apply(len)

    print("\nText length statistics:")
    print(f"Mean: {df['text_length'].mean():.2f}")
    print(f"Median: {df['text_length'].median()}")
    print(f"Min: {df['text_length'].min()}")
    print(f"Max: {df['text_length'].max()}")

    # Plot text length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['text_length'], bins=50)
    plt.title('Text Length Distribution')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('text_length_distribution.png')

    # Analyze most common words
    all_words = ' '.join(df['text']).lower().split()
    word_counts = Counter(all_words)

    print("\nMost common words:")
    for word, count in word_counts.most_common(20):
        print(f"{word}: {count}")

    # Analyze multi-label occurrences
    print("\nMulti-label statistics:")
    df['label_count'] = df[['NORMAL', 'INSULT', 'THREAT', 'OBSCENITY']].sum(axis=1)
    label_count_stats = df['label_count'].value_counts().sort_index()

    for count, num in label_count_stats.items():
        print(f"{count} labels: {num} comments ({num / len(df) * 100:.2f}%)")

    # Plot multi-label distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label_count', data=df)
    plt.title('Number of Labels per Comment')
    plt.xlabel('Number of Labels')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('label_count_distribution.png')

    return df


def prepare_data_for_training(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Prepare data for training by splitting into train, validation, and test sets
    """
    # First split into train+val and test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label_count']  # Stratify by number of labels per comment
    )

    # Then split train+val into train and val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_val_df['label_count']
    )

    print(f"Train set: {len(train_df)} examples")
    print(f"Validation set: {len(val_df)} examples")
    print(f"Test set: {len(test_df)} examples")

    # Save splits to CSV
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)

    return train_df, val_df, test_df


# Example usage
if __name__ == "__main__":
    # Parse and preprocess the dataset
    file_path = "./dataset.txt"  # Update with your actual path
    df = parse_and_preprocess_dataset(file_path, clean=True, remove_stopwords=False)

    # Analyze the dataset
    df = analyze_dataset(df)

    # Prepare data for training
    train_df, val_df, test_df = prepare_data_for_training(df)

    print("Data preprocessing completed!")