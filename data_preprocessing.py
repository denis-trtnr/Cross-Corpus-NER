# data_preprocessing.py

import os
import random
import warnings
from datetime import datetime
from typing import List, Tuple
from collections import Counter

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    DataCollatorForTokenClassification,
)
from adapters import AutoAdapterModel

# Statische Variablen und URLs
BASE_ADAPTER_DIR = "/netscratch/dtrautner/studienarbeit/results"
START_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

SETH_TRAIN_URL = 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/SETH-train.iob'
SETH_TEST_URL = 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/SETH-test.iob'

VARIOME_TRAIN_URL = 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/Variome-train.iob'
VARIOME_TEST_URL = 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/Variome-test.iob'

VARIOME120_TRAIN_URL = 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/Variome120-train.iob'
VARIOME120_TEST_URL = 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/Variome120-test.iob'

AMIA_TRAIN_URL = 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/amia-train.iob'
AMIA_TEST_URL = 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/amia-test.iob'

TMVAR_TRAIN_URL = 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/tmvar-train.iob'
TMVAR_TEST_URL = 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/tmvar-test.iob'


# -------------------------------
# Funktionen zum Herunterladen der Daten
# -------------------------------

def download_data(url: str, filename: str) -> None:
    """Lädt die Daten von der URL herunter und speichert sie lokal."""
    response = requests.get(url)
    response.raise_for_status()  # Überprüfen, ob die Anfrage erfolgreich war
    with open(filename, 'w') as file:
        file.write(response.text)


def download_all_data() -> None:
    """Lädt alle benötigten Datensätze herunter."""
    download_data(SETH_TRAIN_URL, 'SETH-train.iob')
    download_data(SETH_TEST_URL, 'SETH-test.iob')
    download_data(VARIOME_TRAIN_URL, 'Variome-train.iob')
    download_data(VARIOME_TEST_URL, 'Variome-test.iob')
    download_data(VARIOME120_TRAIN_URL, 'Variome120-train.iob')
    download_data(VARIOME120_TEST_URL, 'Variome120-test.iob')
    download_data(AMIA_TRAIN_URL, 'amia-train.iob')
    download_data(AMIA_TEST_URL, 'amia-test.iob')
    download_data(TMVAR_TRAIN_URL, 'tmvar-train.iob')
    download_data(TMVAR_TEST_URL, 'tmvar-test.iob')


# -------------------------------
# Funktionen zum Einlesen und Parsen der Daten
# -------------------------------

def read_data_with_sentences(filename: str) -> List[List[List[Tuple[str, str]]]]:
    """
    Liest die IOB-Datei ein und gibt eine Liste von Dokumenten zurück.
    Jedes Dokument ist eine Liste von Sätzen, und jeder Satz ist eine Liste von (Token, Label)-Tupeln.
    """
    documents = []
    current_document = []
    current_sentence = []
    
    with open(filename, 'r') as file:
        next(file)  # Ignoriere die erste Zeile (Spaltenüberschriften)
        for line in file:
            line = line.strip()
            if not line or line == "Word,Tag" or line == ",O":
                continue
            if line.startswith("#") and "," not in line:
                if current_document:
                    documents.append(current_document)
                current_document = []
            elif line == ",":
                if current_sentence:
                    current_document.append(current_sentence)
                current_sentence = []
            elif line:
                try:
                    token, label = line.rsplit(",", 1)
                    current_sentence.append((token, label))
                except ValueError:
                    print(f"Zeile konnte nicht verarbeitet werden: {line}")
        if current_sentence:
            current_document.append(current_sentence)
        if current_document:
            documents.append(current_document)
    return documents


# Wird erzeit nicht genutzt, nur zum debugging da
def count_sentences_and_entities(title: str, documents: List[List[List[Tuple[str, str]]]]) -> Tuple[int, int]:
    """
    Zählt die Sätze und Entitäten in den Dokumenten.
    Es wird die Anzahl der Sätze und die Anzahl der Entitäten (basierend auf B-/I-Tags) ausgegeben.
    """
    sentence_count = 0
    entity_count = 0
    for doc in documents:
        for sentence in doc:
            sentence_count += 1
            in_entity = False
            for _, label in sentence:
                if label.startswith("B-"):
                    entity_count += 1
                    in_entity = True
                elif label.startswith("I-"):
                    if not in_entity:
                        continue
                else:
                    in_entity = False
    print(f"{title}: {len(documents)} Dokumente, {sentence_count} Sätze, {entity_count} Entitäten")
    return sentence_count, entity_count


# -------------------------------
# Splitten der Daten in Train/Dev
# -------------------------------

def split_documents(documents: List, train_ratio: float = 0.8) -> Tuple[List, List]:
    """Teilt die Dokumente zufällig in Trainings- und Dev-Splits auf."""
    random.shuffle(documents)
    split_index = int(len(documents) * train_ratio)
    return documents[:split_index], documents[split_index:]


# -------------------------------
# Daten vorbereiten für das Training
# -------------------------------

# Zunächst: Laden der Rohdaten aus den IOB-Dateien
def load_all_datasets() -> List[dict]:
    """Lädt alle Datensätze, splittet sie in Train/Dev und gibt eine Liste von Dictionaries zurück."""
    # Download (optional, falls noch nicht vorhanden)
    download_all_data()
    
    datasets = [
        {"name": "SETH", "train": read_data_with_sentences("SETH-train.iob"), "test": read_data_with_sentences("SETH-test.iob")},
        {"name": "Variome", "train": read_data_with_sentences("Variome-train.iob"), "test": read_data_with_sentences("Variome-test.iob")},
        {"name": "Variome120", "train": read_data_with_sentences("Variome120-train.iob"), "test": read_data_with_sentences("Variome120-test.iob")},
        {"name": "Amia", "train": read_data_with_sentences("amia-train.iob"), "test": read_data_with_sentences("amia-test.iob")},
        {"name": "TmVar", "train": read_data_with_sentences("tmvar-train.iob"), "test": read_data_with_sentences("tmvar-test.iob")},
    ]
    
    split_datasets = []
    for data in datasets:
        print(f"Erzeuge Train/Dev-Split für {data['name']}...")
        train_split, dev_split = split_documents(data['train'], train_ratio=0.8)
        split_datasets.append({
            "name": data["name"],
            "train": train_split,
            "dev": dev_split,
            "test": data["test"]
        })
    return split_datasets


# Label Mapping und Vorbereitung der Trainingsdaten
UNIFIED_LABELS = [
    "O",
    "B-Gene/Protein", "I-Gene/Protein",
    "B-SNP", "I-SNP",
    "B-Mutation", "I-Mutation",
    "B-DNA_Mutation", "I-DNA_Mutation",
    "B-RNA_Mutation", "I-RNA_Mutation",
    "B-Protein_Mutation", "I-Protein_Mutation",
    "B-Disease/Disorder", "I-Disease/Disorder",
    "B-Patient_Attribute", "I-Patient_Attribute",
    "B-Phenomena/Concepts", "I-Phenomena/Concepts",
    "B-Locus", "I-Locus",
    "B-Measurement", "I-Measurement",
    "B-Ethnicity", "I-Ethnicity",
]

LABEL_MAPPING = {
    "SETH": {
        "O": "O",
        "B-Gene": "B-Gene/Protein",
        "I-Gene": "I-Gene/Protein",
        "B-RS": "O",
        "I-RS": "O",
        "B-SNP": "B-SNP",
        "I-SNP": "I-SNP"
    },
    "Variome": {
        "O": "O",
        "B-Concepts_Ideas": "B-Phenomena/Concepts",
        "I-Concepts_Ideas": "I-Phenomena/Concepts",
        "B-Disorder": "B-Disease/Disorder",
        "I-Disorder": "I-Disease/Disorder",
        "B-Phenomena": "B-Phenomena/Concepts",
        "I-Phenomena": "I-Phenomena/Concepts",
        "B-Physiology": "B-Phenomena/Concepts",
        "I-Physiology": "I-Phenomena/Concepts",
        "B-age": "B-Patient_Attribute",
        "I-age": "I-Patient_Attribute",
        "B-gender": "B-Patient_Attribute",
        "I-gender": "I-Patient_Attribute",
        "B-body-part": "B-Patient_Attribute",
        "I-body-part": "I-Patient_Attribute",
        "B-cohort-patient": "B-Patient_Attribute",
        "I-cohort-patient": "I-Patient_Attribute",
        "B-disease": "B-Disease/Disorder",
        "I-disease": "I-Disease/Disorder",
        "B-ethnicity": "B-Ethnicity",
        "I-ethnicity": "I-Ethnicity",
        "B-gene": "B-Gene/Protein",
        "I-gene": "I-Gene/Protein",
        "B-mutation": "B-Mutation",
        "I-mutation": "I-Mutation",
        "B-size": "B-Measurement",
        "I-size": "I-Measurement"
    },
    "Variome120": {
        "O": "O",
        "B-mutation": "B-Mutation",
        "I-mutation": "I-Mutation"
    },
    "Amia": {
        "O": "O",
        "B-DNA_Mutation": "B-DNA_Mutation",
        "I-DNA_Mutation": "I-DNA_Mutation",
        "B-DNA_modification": "B-DNA_Mutation",
        "I-DNA_modification": "I-DNA_Mutation",
        "B-Gene_protein": "B-Gene/Protein",
        "I-Gene_protein": "I-Gene/Protein",
        "B-Mutation": "B-Mutation",
        "I-Mutation": "I-Mutation",
        "B-Protein_Mutation": "B-Protein_Mutation",
        "I-Protein_Mutation": "I-Protein_Mutation",
        "B-RNA": "B-RNA_Mutation",
        "I-RNA": "I-RNA_Mutation",
        "B-RNA_Mutation": "B-RNA_Mutation",
        "I-RNA_Mutation": "I-RNA_Mutation",
        "B-dbSNP": "B-SNP",
        "I-dbSNP": "I-SNP",
        "B-locus": "B-Locus",
        "I-locus": "I-Locus"
    },
    "TmVar": {
        "O": "O",
        "B-DNAMutation": "B-DNA_Mutation",
        "I-DNAMutation": "I-DNA_Mutation",
        "B-ProteinMutation": "B-Protein_Mutation",
        "I-ProteinMutation": "I-Protein_Mutation",
        "B-SNP": "B-SNP",
        "I-SNP": "I-SNP"
    }
}

LABEL_TO_ID = {label: idx for idx, label in enumerate(UNIFIED_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

def map_labels(corpus_name: str, raw_label: str) -> str:
    """Mappt ein Rohlabel aus einem bestimmten Korpus auf das vereinheitlichte Label."""
    mapping = LABEL_MAPPING.get(corpus_name, {})
    if raw_label in mapping:
        return mapping[raw_label]
    else:
        warnings.warn(f"Label '{raw_label}' aus Datensatz '{corpus_name}' konnte nicht gemappt werden. Fallback zu 'O'.")
        return "O"

def prepare_data(documents: List, corpus_name: str) -> Dataset:
    """
    Bereitet die Daten für das Training vor, indem die Tokens und die gemappten Labels
    in ein HF Dataset umgewandelt werden.
    """
    sentences = []
    labels = []
    for doc in documents:
        for sentence in doc:
            sentences.append([token for token, _ in sentence])
            labels.append([LABEL_TO_ID[map_labels(corpus_name, label)] for _, label in sentence])
    return Dataset.from_dict({"tokens": sentences, "ner_tags": labels})


def process_all_datasets() -> list:
    """
    Verarbeitet alle Datensätze: Erzeugen von Train/Dev/Test-Splits, Vorbereitung der Daten
    und Erzeugung eines Lists von Dictionaries mit den verarbeiteten Datasets.
    """
    split_datasets = []
    raw_datasets = load_all_datasets()
    for data in raw_datasets:
        print(f"Verarbeite {data['name']}...")
        train_dataset = prepare_data(data["train"], data["name"])
        dev_dataset = prepare_data(data["dev"], data["name"])
        test_dataset = prepare_data(data["test"], data["name"])
        split_datasets.append({
            "name": data["name"],
            "train": train_dataset,
            "dev": dev_dataset,
            "test": test_dataset
        })
    return split_datasets


# -------------------------------
# Tokenisierung und Label Alignment
# -------------------------------

# Hier ein Beispiel mit einem BiomedBERT-Modell; passe ggf. den Pretrained Model-Namen an.
pretrained_model = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
config = AutoConfig.from_pretrained(pretrained_model, num_labels=len(LABEL_TO_ID), label2id=LABEL_TO_ID, id2label=ID_TO_LABEL)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
# Modell wird hier nicht weiter benötigt – wird später im Training geladen (z. B. in train.py oder adapter_composition.py)
model = AutoAdapterModel.from_pretrained(pretrained_model, config=config)


def align_labels(original_labels, word_ids):
    aligned_labels = []
    previous_word_id = None

    for word_idx in word_ids:
        if word_idx != previous_word_id:
            # Beginn eines neuen Wortes oder Special Token
            if word_idx is None:  # Special Token
                aligned_labels.append(-100)
            else:  # Beginn eines neuen Wortes, altes wird gespeichert
                aligned_labels.append(original_labels[word_idx])
            previous_word_id = word_idx
        else:
            if word_idx is None: #Es handelt sich um ein Special token
                aligned_labels.append(-100)
            else:
                # Das Token gehört zum selben Wort wie das vorherige Token
                label = original_labels[word_idx]
                # Wenn das Label B-* ist, ändere es zu I-* (B-* sind ungerade Labels)
                if label % 2 == 1:
                    label += 1
                aligned_labels.append(label)
    return aligned_labels


def tokenize_and_align(inputs: dict) -> dict:
    """
    Tokenisiert die Eingabetexte und ordnet den Token die entsprechenden NER-Labels zu.
    """
    tokenized_inputs = tokenizer(
        inputs["tokens"], truncation=True, is_split_into_words=True
    )
    original_labels = inputs["ner_tags"]
    aligned_labels = []
    for i, labels_per_sentence in enumerate(original_labels):
        word_ids = tokenized_inputs.word_ids(i)
        aligned_labels.append(align_labels(labels_per_sentence, word_ids))
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def tokenize_all_datasets(processed_datasets: list) -> dict:
    """
    Tokenisiert alle verarbeiteten Datasets (Train/Dev/Test) und gibt ein Dictionary zurück,
    in dem die tokenisierten Splits pro Korpus gespeichert sind.
    """
    tokenized_datasets = {}
    for data in processed_datasets:
        print(f"Tokenisiere {data['name']}...")
        train_tokenized = data["train"].map(
            tokenize_and_align,
            batched=True,
            remove_columns=data["train"].column_names
        )
        dev_tokenized = data["dev"].map(
            tokenize_and_align,
            batched=True,
            remove_columns=data["dev"].column_names
        )
        test_tokenized = data["test"].map(
            tokenize_and_align,
            batched=True,
            remove_columns=data["test"].column_names
        )
        tokenized_datasets[data["name"]] = {
            "train": train_tokenized,
            "dev": dev_tokenized,
            "test": test_tokenized
        }
    return tokenized_datasets


# -------------------------------
# Data Collator für Token Classification
# -------------------------------

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# -------------------------------
# Modul-Interface: Funktionen, die von anderen Skripten genutzt werden
# -------------------------------

def get_tokenized_datasets() -> dict:
    """
    Führt die komplette Pipeline: Rohdaten laden, aufteilen, vorbereiten, tokenisieren
    und liefert ein Dictionary der tokenisierten Datasets.
    """
    processed = process_all_datasets()
    tokenized = tokenize_all_datasets(processed)
    return tokenized


# Zum Testen, wenn das Modul direkt ausgeführt wird.
if __name__ == "__main__":
    # Lade alle tokenisierten Datasets und zeige einen Überblick
    tokenized_datasets = get_tokenized_datasets()
    print("Verfügbare Datasets:")
    for name, splits in tokenized_datasets.items():
        print(f"- {name}: Train={len(splits['train'])}, Dev={len(splits['dev'])}, Test={len(splits['test'])}")
