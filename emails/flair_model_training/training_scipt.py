import quopri
import pandas as pd
from text_cleaning import fetch_body, preprocess_text
from sklearn.model_selection import train_test_split
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings


# Initial Data Processing Functions


def decode_quoted_printable(text: str) -> str: 
    """
    Decodes incorrectly encoded emails using ISO 8859-1.
    
    Args: 
        text (str): Email, stored as a string in a .csv file, column 'Content'.
    
    Returns: 
        text (str): Email string decoded using iso 8859-1. 

    Examples: 
        >> encoded_text = "=F6ffentlichen"
        >> decoded_text = quopri.decodestring(encoded_text).decode('iso-8859-1')
        "öffentlichen"
    """
    if isinstance(text, str):
        try:
            decoded = quopri.decodestring(text.encode("latin1")).decode("iso-8859-1")
            return decoded
        except Exception as e:
            return str(e)
    return text


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the DataFrame: unnecessary column removal, emails normalization.
    
    Args: 
        df (pd.DataFrame): unprocessed DataFrame read from .csv file (pd.read_csv).
        
    Returns: 
        df (pd.DataFrame): DataFrame with clean emails.
        
    """
    df = df.drop(["Unnamed: 0"], axis=1)
    df["Content_fixed"] = df["Content"].apply(decode_quoted_printable)
    df = duplicate_rows(df)
    df = df.dropna()
    df["Email_body"] = df["Content_fixed"].apply(fetch_body)
    df["Processed_texts"] = df["Email_body"].apply(preprocess_text)
    return df


def duplicate_rows(df: pd.DataFrame):
    """
    Duplicates rows with 'Category' as 'PRAEVENTION'.
    This category has only one email, and sklearnn doesn't allow 
    train/test split unless there are at least 2 data entries per category
    
    Args:
        df (pd.DataFrame): Dataframe to process. 
        
    Returns:
        df (pd.DataFrame): Dataframe with 2 entries of category 'PRAEVENTION'.
    """
    condition = df["Category"] == "PRAEVENTION"
    rows_to_duplicate = df[condition]
    duplicated_rows = rows_to_duplicate.copy()
    df = pd.concat([df, duplicated_rows], ignore_index=True)
    return df


# Creation of DataFrame for training with Flair


def create_flair_dataset(df: pd.DataFrame):
    """
    Create Flair-compatible dataset.
    
    Args:
        df (pd.DataFrame): Dataframe with emails.

    Returns: 
        df (pd.DataFrame): Dataframe with 2 new columns:
            'label' contains category names represented numerically with __label__ keyword:
                '__label__14' 
            'label_text' contains emails text preceded with the corresponding label:
                '__label__14 bekannt gehören angehoerig oeffent...'
    """
    
    df["label"] = "__label__" + df["Category"].astype("category").cat.codes.astype(str)
    df["label_text"] = df["label"] + " " + df["Processed_texts"]
    df["label_text"] = df["label_text"].str.rstrip()
    return df


def split_dataset(df: pd.DataFrame):
    """
    Splits dataset into train, test, and validation sets using train_test_split function of sklearn and oversamples the training set.
    
    Args:
        df (pd.DataFrame): Flair-compatible dataset to split.
        
    Returns:
        train, test, valid as training, testing and validation sets (pd.DataFrame)
        num_to_word (dict): numerically encoded categories are keys, category titles as values: 
            {'__label__1': 'AKTIVIERUNG_SIM', ...}
    """
    train, test = train_test_split(
        df, test_size=0.2, random_state=1, stratify=df["label"]
    )
    train, valid = train_test_split(
        train, test_size=0.2, random_state=1, stratify=train["label"]
    )

    # Oversample dataset
    max_size = train["Category"].value_counts().max()
    lst = [train]
    for class_index, group in train.groupby("Category"):
        lst.append(group.sample(max_size - len(group), replace=True))

    train = pd.concat(lst)
    
    num_to_word = train.set_index('label')['Category'].to_dict()
    
    return train, test, valid, num_to_word


def save_data(train, test, valid):
    """
    Save train, valid and test files as train.txt, valid.txt, and test.txt as Flair requires.
    Specify column which contains texts to train the model on (label_text)
    
    Args:
        train, test, valid (pd.DataFrame)
        
    """
    train.to_csv(
        "flair/data/train.txt", columns=["label_text"], index=False, header=False
    )
    valid.to_csv(
        "flair/data/valid.txt", columns=["label_text"], index=False, header=False
    )
    test.to_csv(
        "flair/data/test.txt", columns=["label_text"], index=False, header=False
    )


# Training Flair Model


def create_flair_corpus():
    """
    Create Flair Corpus from dataset files (train.txt, valid.txt, test.txt).
    
    Returns:
        corpus (flair.datasets.ClassificationCorpus)
        label_type (str)
    """
    data_folder = "flair/data/"
    label_type = "label"

    corpus: Corpus = ClassificationCorpus(
        data_folder,
        test_file="test.txt",
        dev_file="valid.txt",
        train_file="train.txt",
        label_type=label_type,
    )

    return corpus, label_type


def define_classifier(corpus, label_type):
    """
    Define Flair TextClassifier and Trainer.
    
    Args:
        corpus (flair.datasets.ClassificationCorpus)
        label_type (str)
    
    Returns:
        trainer (flair.trainers.ModelTrainer)
    
    """
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    document_embeddings = TransformerDocumentEmbeddings(
        "bert-base-german-cased", fine_tune=True
    )
    classifier = TextClassifier(
        document_embeddings, label_dictionary=label_dict, label_type=label_type
    )
    trainer = ModelTrainer(classifier, corpus)
    return trainer


def train_model(trainer):
    """
    Train the Flair model.
    
    Args:
        trainer (flair.trainers.ModelTrainer):
    
    Returns:
        Automatically stores the best and final models in flair/model/ directory.
    """
    trainer.train(
        "flair/model/",
        embeddings_storage_mode="gpu",
        learning_rate=0.005,
        mini_batch_size=16,
        mini_batch_chunk_size=4,
        # sampler=ImbalancedClassificationDatasetSampler,
        max_epochs=10,
    )


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("full_dataset.csv")

    # Process data
    df_processed = process_data(df)

    # Create Flair-compatible dataset
    flair_dataset = create_flair_dataset(df_processed)

    # Split dataset
    train_data, test_data, valid_data, num_to_word = split_dataset(flair_dataset)

    # Save dataset files
    save_data(train_data, test_data, valid_data)

    # Create Flair Corpus
    flair_corpus, label_type = create_flair_corpus()

    # Define Classifier
    trainer = define_classifier(flair_corpus, label_type)

    # Train model
    train_model(trainer)
