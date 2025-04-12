import os
import random
from typing import List, Tuple

import requests
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    DataCollatorForTokenClassification,
)
from mapping_utils import (
    UNIFIED_LABELS_GRANULAR, 
    UNIFIED_LABELS_BROAD, 
    map_labels
)
from adapters import AutoAdapterModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class DataPreprocessor:
    def __init__(self, 
                 mapping_type: str = "granular", 
                 base_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 data_dir: str = "data"):
        
        self.mapping_type = mapping_type
        self.base_model = base_model
        self.data_dir = os.path.abspath(os.path.join(BASE_DIR, "..", data_dir))

        # Setze die globalen Label-Variablen in der Instanz
        if self.mapping_type == "granular":
            self.unified_labels = UNIFIED_LABELS_GRANULAR
        else:
            self.unified_labels = UNIFIED_LABELS_BROAD
        self.label_to_id = {label: idx for idx, label in enumerate(self.unified_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        # Initialisiere Modellkomponenten
        self.config = AutoConfig.from_pretrained(self.base_model, 
                                                 num_labels=len(self.unified_labels),
                                                 label2id=self.label_to_id, 
                                                 id2label=self.id_to_label)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoAdapterModel.from_pretrained(self.base_model, config=self.config)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    # -------------------------------
    # Funktionen zum Herunterladen der Daten
    # -------------------------------

    def download_data(self, filename: str, url: str) -> None:
        """Lädt die Daten von der URL herunter und speichert sie lokal."""
        os.makedirs(self.data_dir, exist_ok=True)
        file_path = os.path.join(self.data_dir, filename)
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, 'w') as file:
            file.write(response.text)



    def download_all_data(self) -> None:
        """Lädt alle benötigten Datensätze herunter."""
        urls = {
            'SETH-train.iob': 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/SETH-train.iob',
            'SETH-test.iob': 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/SETH-test.iob',
            'Variome-train.iob': 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/Variome-train.iob',
            'Variome-test.iob': 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/Variome-test.iob',
            'Variome120-train.iob': 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/Variome120-train.iob',
            'Variome120-test.iob': 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/Variome120-test.iob',
            'amia-train.iob': 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/amia-train.iob',
            'amia-test.iob': 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/amia-test.iob',
            'tmvar-train.iob': 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/tmvar-train.iob',
            'tmvar-test.iob': 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/tmvar-test.iob',
        }
        for filename, url in urls.items():
            self.download_data(filename, url)


    # -------------------------------
    # Funktionen zum Einlesen und Parsen der Daten
    # -------------------------------

    def read_data_with_sentences(self, filepath: str) -> List[List[List[Tuple[str, str]]]]:
        """
        Liest die IOB-Datei ein und gibt eine Liste von Dokumenten zurück.
        Jedes Dokument ist eine Liste von Sätzen, und jeder Satz ist eine Liste von (Token, Label)-Tupeln.
        """
        documents = []
        current_document = []
        current_sentence = []
        
        with open(filepath, 'r') as file:
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

    def split_documents(self, documents: List, train_ratio: float) -> Tuple[List, List]:
        """Teilt die Dokumente zufällig in Trainings- und Dev-Splits auf."""
        random.shuffle(documents)
        split_index = int(len(documents) * train_ratio)
        return documents[:split_index], documents[split_index:]


    # -------------------------------
    # Daten vorbereiten für das Training
    # -------------------------------

    # Zunächst: Laden der Rohdaten aus den IOB-Dateien
    def load_all_datasets(self) -> List[dict]:
        """Lädt alle Datensätze, splittet sie in Train/Dev und gibt eine Liste von Dictionaries zurück."""
        # Download (optional, falls noch nicht vorhanden)
        self.download_all_data()
        
        datasets = [
            {
                "name": "SETH", 
                "train": self.read_data_with_sentences(os.path.join(self.data_dir, "SETH-train.iob")), 
                "test": self.read_data_with_sentences(os.path.join(self.data_dir, "SETH-test.iob"))
            },
            {
                "name": "Variome", 
                "train": self.read_data_with_sentences(os.path.join(self.data_dir, "Variome-train.iob")), 
                "test": self.read_data_with_sentences(os.path.join(self.data_dir, "Variome-test.iob"))
            },
            {
                "name": "Variome120",
                "train": self.read_data_with_sentences(os.path.join(self.data_dir, "Variome120-train.iob")), 
                "test": self.read_data_with_sentences(os.path.join(self.data_dir, "Variome120-test.iob"))
            },
            {
                "name": "Amia",
                "train": self.read_data_with_sentences(os.path.join(self.data_dir, "amia-train.iob")), 
                "test": self.read_data_with_sentences(os.path.join(self.data_dir, "amia-test.iob"))
            },
            {
                "name": "TmVar",
                "train": self.read_data_with_sentences(os.path.join(self.data_dir, "tmvar-train.iob")), 
                "test": self.read_data_with_sentences(os.path.join(self.data_dir, "tmvar-test.iob"))
            },
        ]
        
        split_datasets = []
        for data in datasets:
            print(f"Erzeuge Train/Dev-Split für {data['name']}...")
            train_split, dev_split = self.split_documents(data['train'], train_ratio=0.8)
            split_datasets.append({
                "name": data["name"],
                "train": train_split,
                "dev": dev_split,
                "test": data["test"]
            })
        return split_datasets

    def prepare_data(self, documents: List, corpus_name: str) -> Dataset:
        """
        Bereitet die Daten für das Training vor, indem die Tokens und die gemappten Labels
        in ein HF Dataset umgewandelt werden.
        """
        sentences = []
        labels = []
        for doc in documents:
            for sentence in doc:
                sentences.append([token for token, _ in sentence])
                labels.append([self.label_to_id[map_labels(corpus_name, label, self.mapping_type)]
                               for _, label in sentence])
        return Dataset.from_dict({"tokens": sentences, "ner_tags": labels})


    def process_all_datasets(self) -> list:
        """
        Verarbeitet alle Datensätze: Erzeugen von Train/Dev/Test-Splits, Vorbereitung der Daten
        und Erzeugung eines Lists von Dictionaries mit den verarbeiteten Datasets.
        """
        processed_datasets = []
        raw_datasets = self.load_all_datasets()
        for data in raw_datasets:
            print(f"Verarbeite {data['name']}...")
            train_dataset = self.prepare_data(data["train"], data["name"])
            dev_dataset = self.prepare_data(data["dev"], data["name"])
            test_dataset = self.prepare_data(data["test"], data["name"])
            processed_datasets.append({
                "name": data["name"],
                "train": train_dataset,
                "dev": dev_dataset,
                "test": test_dataset
            })
        return processed_datasets


    # -------------------------------
    # Tokenisierung und Label Alignment
    # -------------------------------


    def align_labels(self, original_labels, word_ids):
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


    def tokenize_and_align(self, inputs: dict) -> dict:
        """
        Tokenisiert die Eingabetexte und ordnet den Token die entsprechenden NER-Labels zu.
        """
        tokenized_inputs = self.tokenizer(
            inputs["tokens"], truncation=True, is_split_into_words=True
        )
        original_labels = inputs["ner_tags"]
        aligned_labels = []
        for i, labels_per_sentence in enumerate(original_labels):
            word_ids = tokenized_inputs.word_ids(i)
            aligned_labels.append(self.align_labels(labels_per_sentence, word_ids))
        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs


    def tokenize_all_datasets(self, processed_datasets: list) -> dict:
        """
        Tokenisiert alle verarbeiteten Datasets (Train/Dev/Test) und gibt ein Dictionary zurück,
        in dem die tokenisierten Splits pro Korpus gespeichert sind.
        """
        tokenized_datasets = {}
        for data in processed_datasets:
            print(f"Tokenisiere {data['name']}...")
            train_tokenized = data["train"].map(
                self.tokenize_and_align,
                batched=True,
                remove_columns=data["train"].column_names
            )
            dev_tokenized = data["dev"].map(
                self.tokenize_and_align,
                batched=True,
                remove_columns=data["dev"].column_names
            )
            test_tokenized = data["test"].map(
                self.tokenize_and_align,
                batched=True,
                remove_columns=data["test"].column_names
            )
            tokenized_datasets[data["name"]] = {
                "train": train_tokenized,
                "dev": dev_tokenized,
                "test": test_tokenized
            }
        return tokenized_datasets


    def get_tokenized_datasets(self) -> dict:
        """
        Führt die komplette Pipeline aus:
        Daten herunterladen, laden, verarbeiten und tokenisieren.
        Gibt ein Dictionary der tokenisierten Datasets zurück.
        """
        processed = self.process_all_datasets()
        tokenized = self.tokenize_all_datasets(processed)
        return tokenized


# Beispielhafte Nutzung, wenn dieses Modul direkt ausgeführt wird.
if __name__ == "__main__":
    preprocessor = DataPreprocessor(mapping_type="broad",
                                    base_model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                                    data_dir="data")
    tokenized_datasets = preprocessor.get_tokenized_datasets()
    print("Verfügbare Datasets:")
    for name, splits in tokenized_datasets.items():
        print(f"- {name}: Train={len(splits['train'])}, Dev={len(splits['dev'])}, Test={len(splits['test'])}")
