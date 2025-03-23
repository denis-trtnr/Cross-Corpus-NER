import requests
import os
import random
import warnings
import evaluate
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tabulate import tabulate
from typing import List, Tuple
from collections import Counter
from sklearn.metrics import confusion_matrix
from datasets import Dataset, DatasetDict, concatenate_datasets
from collections import Counter
from datetime import datetime
from matplotlib import cm
from transformers import (
    DistilBertTokenizerFast,
    AutoTokenizer, 
    AutoConfig,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments
)
from adapters import AutoAdapterModel
from adapters import AdapterTrainer, AdapterConfig
from adapters import init
from adapters.composition import Stack
from adapters.composition import Fuse


# 1.) Laden der Daten

#Statische Variablen
BASE_ADAPTER_DIR = "/netscratch/dtrautner/studienarbeit/results"
# BASE_ADAPTER_DIR = "./"  # (Alternative: lokales Verzeichnis)

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

START_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Funktion, um die Daten von von Github herunterzuladen
def download_data(url, filename):
    response = requests.get(url)
    response.raise_for_status()  # Überprüfen, ob die Anfrage erfolgreich war
    with open(filename, 'w') as file:
        file.write(response.text)

# Herunterladen der Daten
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

# Funktion zum parsen der heruntergeladenen Daten
def read_data_with_sentences(filename):
    documents = []
    current_document = []
    current_sentence = []
    
    with open(filename, 'r') as file:
        next(file)  # Ignoriere die erste Zeile (Spaltenüberschriften)
        for line in file:
            line = line.strip()

            # Überspringe header oder leere Zeilen
            if not line or line == "Word,Tag" or line == ",O":
                continue
            
            # Prüfe, ob es eine Dokumenten-ID ist (beginnt mit "#" und hat kein "," bzw. Zeile != "#,O")
            if line.startswith("#") and "," not in line:  
                if current_document:  # Speichere aktuelles Dokument sofern vorhanden
                    documents.append(current_document)
                current_document = []  # Neues Dokument wird gestartet

            # Prüfe, ob es sich um das Satzende handelt.
            elif line == ",":  
                if current_sentence:  # Speichere aktuellen Satz sofern vorhanden
                    current_document.append(current_sentence)
                current_sentence = []  # Neuer Satz wird gestartet
            
            # Token und Label verarbeiten
            elif line:  
                try:
                    # Trenne Token und Label anhand des letzten Kommas (rsplit)
                    # Kann auch mehrere komma geben (z.B. ",,O")
                    token, label = line.rsplit(",", 1)
                    current_sentence.append((token, label))
                except ValueError:
                    print(f"Zeile konnte nicht verarbeitet werden: {line}")
        
        # Letzten Satz und Dokument speichern
        if current_sentence:  # Letzter Satz
            current_document.append(current_sentence)
        if current_document:  # Letztes Dokument
            documents.append(current_document)
    
    return documents

# # 2.) Experiment vorbereiten


# Funktion zum Zählen von Sätzen und Entitäten
def count_sentences_and_entities(title, documents: List[List[List[Tuple[str, str]]]]) -> Tuple[int, int]:
    sentence_count = 0
    entity_count = 0
    
    for doc in documents:
        for sentence in doc:
            sentence_count += 1
            in_entity = False  # Status, ob wir in einer Entität sind
            
            """
            Hier hab ich 2 Möglichkeiten gesehen wie man Entitäten zählen kann
            1. Anzahl der Klassen != O
            2. B-* und I-* Tags zu einer Entität zusammenzählen

            Habe mich für den 2. Weg entscheiden, da sich dies realitätsnaher anfühlte
            """
            for _, label in sentence:
                if label.startswith("B-"):  # Beginn einer neuen Entität
                    entity_count += 1
                    in_entity = True  # Wir sind jetzt in einer Entität
                elif label.startswith("I-"):  # Innerhalb einer Entität
                    if not in_entity:  # Wenn kein "B-" Tag vorher war, ignorieren
                        continue
                else:
                    in_entity = False  # Nicht mehr in einer Entität
    
    print(f"{title}: {len(documents)} Dokumente, {sentence_count} Sätze, {entity_count} Entitäten")




datasets = [
    {"name": "SETH", "train": read_data_with_sentences("SETH-train.iob"), "test": read_data_with_sentences("SETH-test.iob")},
    {"name": "Variome", "train": read_data_with_sentences("Variome-train.iob"), "test": read_data_with_sentences("Variome-test.iob")},
    {"name": "Variome120", "train": read_data_with_sentences("Variome120-train.iob"), "test": read_data_with_sentences("Variome120-test.iob")},
    {"name": "Amia", "train": read_data_with_sentences("amia-train.iob"), "test": read_data_with_sentences("amia-test.iob")},
    {"name": "TmVar", "train": read_data_with_sentences("tmvar-train.iob"), "test": read_data_with_sentences("tmvar-test.iob")},
]

# Iteriere über die Datensätze und gebe Anzahl Sätze und Entities aus
for dataset in datasets:
    count_sentences_and_entities(
        title=f"{dataset['name']} Trainingsdaten", 
        documents=dataset["train"]
    )
    count_sentences_and_entities(
        title=f"{dataset['name']} Testdaten", 
        documents=dataset["test"]
    )




"""
# Zähle und plotte die Klassenverteilung
def count_and_plot_classes(documents: List[List[List[Tuple[str, str]]]], ax, all_classes=None, color=None) -> None:
    labels = []
    for doc in documents:
        for sentence in doc:
            labels.extend(label for _, label in sentence)
    
    # Berechne die Klassenverteilung
    class_distribution = Counter(labels)
    
    # Falls `all_classes` nicht angegeben ist, ermitteln wir sie dynamisch
    if all_classes is None:
        all_classes = sorted(class_distribution.keys())  # Alphabetisch sortieren
    
    # Stelle sicher, dass alle gewünschten Klassen angezeigt werden (auch wenn sie nicht im aktuellen Datensatz vorkommen)
    for cls in all_classes:
        if cls not in class_distribution:
            class_distribution[cls] = 0  # Füge fehlende Klassen mit 0 hinzu
    
    # Sortiere die Klassen nach `all_classes`
    ordered_keys = all_classes
    ordered_values = [class_distribution[cls] for cls in ordered_keys]

    # Plot der Klassenverteilung
    bars = ax.bar(ordered_keys, ordered_values, color=color)
    
    # Absolute und prozentuale Werte hinzufügen
    total_count = sum(class_distribution.values())
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height} ({height / total_count * 100:.1f}%)', 
                     xy=(bar.get_x() + bar.get_width() / 2, height), 
                     ha='center', va='bottom')
    
    ax.set_xlabel('Klassen')
    ax.set_ylabel('Anzahl')




# Farben für die Datensätze aus einer Farbpalette
colors = cm.tab10(range(len(datasets)))  # Unterschiedliche Farben für die Datensätze

# Diagramm für alle Datensätze
fig, axs = plt.subplots(len(datasets) * 2, 1, figsize=(20, 6 * len(datasets) * 2))  # Eine Zeile pro Diagramm

for i, (dataset, color) in enumerate(zip(datasets, colors)):
    # Train split
    count_and_plot_classes(dataset["train"], axs[i * 2], color=color)
    axs[i * 2].set_title(f'{dataset["name"]} - Train')
    axs[i * 2].tick_params(axis='x', rotation=45)  # Drehe die x-Beschriftungen
    
    # Test split
    count_and_plot_classes(dataset["test"], axs[i * 2 + 1], color=color)
    axs[i * 2 + 1].set_title(f'{dataset["name"]} - Test')
    axs[i * 2 + 1].tick_params(axis='x', rotation=45)  # Drehe die x-Beschriftungen

plt.tight_layout()
plt.show()
"""

# Funktion zur Aufteilung in Train/Dev
def split_documents(documents, train_ratio=0.8):
    random.shuffle(documents)  # Zufällige Reihenfolge der Dokumente
    split_index = int(len(documents) * train_ratio)
    train_split = documents[:split_index]
    dev_split = documents[split_index:]
    return train_split, dev_split

# Aktualisiere die Datasets-Liste mit Train/Dev-Splits
split_datasets = []

for data in datasets:
    print(f"Erzeuge Train/Dev-Split für {data['name']}...")
    
    # Train/Dev-Split durchführen
    train_split, dev_split = split_documents(data['train'], train_ratio=0.8)
    
    # Ergebnisse speichern
    split_datasets.append({
        "name": data["name"],
        "train": train_split,
        "dev": dev_split,
        "test": data["test"]
    })

# %%
# Brauchen gemappte Labels, sonst zu wenig Überschneidung:
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

#Mapping-Funktion
def map_labels(corpus_name, raw_label):
    mapping = LABEL_MAPPING.get(corpus_name, {})
    if raw_label in mapping:
        return mapping[raw_label]
    else:
        warnings.warn(f"Label '{raw_label}' aus Datensatz '{corpus_name}' konnte nicht gemappt werden. Fallback zu 'O'.")
        return "O"

# # 3.) Trainieren Sie einen Transformer zur Eigennamenerkennung
#Daten für Training vorbereiten und in die richtige Form bringen
def prepare_data(documents, corpus_name):
    sentences = []
    labels = []
    for doc in documents:
        for sentence in doc:
            sentences.append([token for token, _ in sentence])
            labels.append([
                LABEL_TO_ID[map_labels(corpus_name, label)] for _, label in sentence
            ])
   
    return Dataset.from_dict({"tokens": sentences, "ner_tags": labels})

# Wende prepare_data auf die train, dev und test-Splits an
processed_datasets = []

for data in split_datasets:
    print(f"Verarbeite {data['name']}...")

    # Trainingsdaten vorbereiten
    train_dataset = prepare_data(data["train"], data["name"])

    # Dev-Daten vorbereiten
    dev_dataset = prepare_data(data["dev"], data["name"])

    # Testdaten vorbereiten
    test_dataset = prepare_data(data["test"], data["name"])

    # Speichern der verarbeiteten Datasets
    processed_datasets.append({
        "name": data["name"],
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset
    })


# Tokenizer initialisieren 
pretrained_model = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" #earlier bert-base-uncased UNBEDINGT pubMEDBERT!!!
#tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model, clean_up_tokenization_spaces=True)
config = AutoConfig.from_pretrained(pretrained_model, num_labels=len(LABEL_TO_ID), label2id=LABEL_TO_ID, id2label=ID_TO_LABEL)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
#model = AutoModelForTokenClassification.from_pretrained(pretrained_model, config=config)
# Enable adapter support
#init(model)
model = AutoAdapterModel.from_pretrained(pretrained_model, config=config)


"""
Funktion ordnet NER-Labels den tokenisierten Eigaben basierend auf Wort-IDs zu.
Notwendig, da Tokenizer Wörter in mehrere Tokenaufteilen.

Tokens die beim Training nicht berücksichtigt werden soll, wird der Wert -100 zugewiesen
"""
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


#Funktion tokenisiert die Eingabetexte und ordnet entsprechende Labels zu
def tokenize_and_allign(inputs):
    tokenized_inputs = tokenizer(
        inputs["tokens"], truncation=True, is_split_into_words=True
    )

    original_labels = inputs["ner_tags"] # Originallabels aus den Eingabedaten
    aligned_labels = []
    for i, labels_per_sentence in enumerate(original_labels):
        word_ids = tokenized_inputs.word_ids(i)
        aligned_labels.append(align_labels(labels_per_sentence, word_ids))

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


# Tokenisierte Datensätze erstellen
tokenized_datasets = {}

for data in processed_datasets:
    print(f"Tokenisiere {data['name']}...")

    # Tokenisiere und aligniere die train-, dev- und test-Splits
    train_tokenized = data["train"].map(
        tokenize_and_allign,
        batched=True,
        remove_columns=data["train"].column_names
    )
    dev_tokenized = data["dev"].map(
        tokenize_and_allign,
        batched=True,
        remove_columns=data["dev"].column_names
    )
    test_tokenized = data["test"].map(
        tokenize_and_allign,
        batched=True,
        remove_columns=data["test"].column_names
    )

    # Speichere die tokenisierten Datensätze in einem Dictionary
    tokenized_datasets[data["name"]] = {
        "train": train_tokenized,
        "dev": dev_tokenized,
        "test": test_tokenized
    }

#Data Collator stellt sicher, dass Daten in richtigem Format für Training vorliegen
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


#Prepare Metrics
metric = evaluate.load("seqeval")

def calculate_metrics(predictions_and_labels):
    logits, labels = predictions_and_labels
    predicted_labels  = np.argmax(logits, axis=-1)

    
    true_labels = []
    true_predictions = []

    # Entferne special tokens und konvertierte von IDs zu labels
    for sentence_labels, sentence_predictions in zip(labels, predicted_labels):
        filtered_true_labels = []
        filtered_predictions = []

        for label, prediction in zip(sentence_labels, sentence_predictions):
            if label != -100:  # Ignoriere Special Tokens
                filtered_true_labels.append(ID_TO_LABEL[label])
                filtered_predictions.append(ID_TO_LABEL[prediction])

        if filtered_true_labels:
            true_labels.append(filtered_true_labels)
            true_predictions.append(filtered_predictions)


    # Debug-Ausgabe: Zeige die ersten X Sätze der wahren und vorhergesagten Labels
    """
    x = 5
    print("\n=== DEBUGGING ===")
    for i in range(min(x, len(true_labels))):  # Nur die ersten X Sätze zeigen
        print(f"\nSatz {i+1} - Wahre Labels:")
        print(true_labels[i][:10])  # Zeige die ersten 10 Labels des Satzes
        print(f"Satz {i+1} - Vorhergesagte Labels:")
        print(true_predictions[i][:10])
    print("=================")
    """


    overall_metrics = metric.compute(predictions=true_predictions, references=true_labels, zero_division=1)

    # Extrahiere die allgemeinen Metriken
    overall_result = {
        "accuracy": overall_metrics["overall_accuracy"],
        "f1": overall_metrics["overall_f1"],
        "precision": overall_metrics["overall_precision"],
        "recall": overall_metrics["overall_recall"],
    }
    """
    wandb.log({
        "eval_accuracy": overall_metrics["overall_accuracy"],
        "eval_f1": overall_metrics["overall_f1"],
        "eval_precision": overall_metrics["overall_precision"],
        "eval_recall": overall_metrics["overall_recall"],
    })
    """
    return overall_result

# ## 3.1) Trainieren der Modelle

#model = AutoAdapterModel.from_pretrained(pretrained_model)
#model = AutoModelForTokenClassification.from_pretrained(
#        pretrained_model,
#        id2label=ID_TO_LABEL,
#        label2id=LABEL_TO_ID,
#    )

#init(model)

# Liste zum reinspeichern der Adapter
trained_adapters = []

# Dictionary zur Speicherung der Trainer pro Adapter
adapter_trainers = {} 

# Evaluation und Zusammenfassung aller Modelle
results_summary = []


# Funktion zum Trainieren mit Adapter
def train_with_adapter(dataset_name, tokenized_data):
    # Konfiguriere den Adapter
    # ✅ Reset W&B environment variables to force a truly separate run
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)

    wandb.init(
        project="CrossCorpusNER",   # Dein Projektname in W&B
        name=f"Train_{dataset_name}_{wandb.util.generate_id()}",  # z.B. Laufname = Name des Datensatzes
        tags=[f"timestamp_{START_TIME}", dataset_name],
    )

    lr = wandb.config.learning_rate if hasattr(wandb.config, "learning_rate") else 2e-4
    batch_size = wandb.config.batch_size if hasattr(wandb.config, "batch_size") else 8
    epochs = wandb.config.num_train_epochs if hasattr(wandb.config, "num_train_epochs") else 1

    adapter_config = AdapterConfig.load(
        "houlsby",  # Standard-Adapter-Konfiguration, kann angepasst werden
    )
    adapter_name = f"{dataset_name}_adapter"
    print(f"Erstellung des Adapters {adapter_name}")
    model.add_adapter(f"{adapter_name}", config=adapter_config)
    model.add_tagging_head(f"{adapter_name}", num_labels=len(ID_TO_LABEL), id2label=ID_TO_LABEL)
    print(f"Adapter erstellt: {adapter_name}")

    model.set_active_adapters(Stack(adapter_name))
    # Hauptmodell einfrieren und nur Adapter trainieren
    model.train_adapter(adapter_name)

    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_ADAPTER_DIR,"results",dataset_name),
        eval_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=f"./logs/{dataset_name}",
        logging_strategy="epoch",
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        report_to="wandb",
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["dev"],
        tokenizer=tokenizer,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=calculate_metrics,
    )

    trainer.train()
    trainer.evaluate()

    adapter_path = os.path.join(BASE_ADAPTER_DIR,"adapters",adapter_name)
    head_path = os.path.join(BASE_ADAPTER_DIR,"heads", f"{adapter_name}")

    model.save_adapter(adapter_path, adapter_name)
    model.save_head(head_path, f"{adapter_name}")
    trained_adapters.append(adapter_name)
    adapter_trainers[adapter_name] = trainer
    print(model.adapter_summary())

    for test_dataset_name, dataset in tokenized_datasets.items():
        print(f"Testen auf {test_dataset_name}-Testset...")

        # Predictions auf dem Testset
        predictions, labels, _ = trainer.predict(dataset["test"])
        predictions = np.argmax(predictions, axis=-1)

        # Bereite die Labels und Vorhersagen zur Auswertung vor
        # Bereite die Labels und Vorhersagen zur Auswertung vor
        true_labels = [[UNIFIED_LABELS[l] for l in label if l != -100] for label in labels]
        pred_labels = [[UNIFIED_LABELS[p] for (p, l) in zip(pred, label) if l != -100] 
                    for pred, label in zip(predictions, labels)]

        # Berechne seqeval-Metriken
        overall_metrics = metric.compute(predictions=pred_labels, references=true_labels, zero_division=1)

        # Flache die Labels und Vorhersagen für die Confusion-Matrix
        true_labels_flat = [label for sublist in true_labels for label in sublist]
        pred_labels_flat = [pred for sublist in pred_labels for pred in sublist]

        # Reduziere die Klassen auf tatsächlich vorkommende Labels
        unique_true_labels = set(true_labels_flat)
        unique_pred_labels = set(pred_labels_flat)
        relevant_classes = sorted(unique_true_labels.union(unique_pred_labels))

        # Berechne die Confusion-Matrix für relevante Klassen
        cm = confusion_matrix(true_labels_flat, pred_labels_flat, labels=relevant_classes)

        # Speichere die Ergebnisse für die Zusammenfassung
        results_summary.append({
            "Adapter": adapter_name,
            "Tested_on": test_dataset_name,
            "accuracy": overall_metrics["overall_accuracy"],
            "f1": overall_metrics["overall_f1"],
            "precision": overall_metrics["overall_precision"],
            "recall": overall_metrics["overall_recall"],
            "confusion_matrix": cm
        })
        """
        wandb.log({
            "test_accuracy": overall_metrics["overall_accuracy"],
            "test_f1": overall_metrics["overall_f1"],
            "test_precision": overall_metrics["overall_precision"],
            "test_recall": overall_metrics["overall_recall"],
        })
        """


        # Berechne die prozentuale Confusion-Matrix
        # Entferne weile Grafisch
        """
        cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100
        annotations = np.array([
            ["{:.2f}%\n({})".format(cm_percentage[i, j], cm[i, j])
                for j in range(len(relevant_classes))] for i in range(len(relevant_classes))
        ])

        # Visualisiere die Confusion-Matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_percentage, annot=annotations, fmt='', 
                    xticklabels=relevant_classes, yticklabels=relevant_classes, cmap='Blues')

        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for Adapter {adapter_name} tested on {test_dataset_name}')
        plt.show()
        """
    wandb.finish()
    model.set_active_adapters(None)
    


# Training starten
for dataset_name, tokenized_data in tokenized_datasets.items():
    print(f"Starte Finetuning für {dataset_name}...")
    train_with_adapter(dataset_name, tokenized_data)

def activate_adapter(adapter_name):
    print(f"Aktiviere Adapter: {adapter_name}")
    model.set_active_adapters(Stack(adapter_name))

# Evaluation und Zusammenfassung aller Modelle
#results_summary = []
"""
def evaluate_all_adapters():
    for adapter_name in trained_adapters:
        print(f"\nEvaluierung des Adapters {adapter_name}...")
        activate_adapter(adapter_name)
        #trainer = adapter_trainers[adapter_name]  # Hole den spezifischen Trainer für den Adapter

        eval_trainer = adapter_trainers[adapter_name]


        for test_dataset_name, dataset in tokenized_datasets.items():
            print(f"Testen auf {test_dataset_name}-Testset...")

            # Predictions auf dem Testset
            predictions, labels, _ = eval_trainer.predict(dataset["test"])
            predictions = np.argmax(predictions, axis=-1)

            # Bereite die Labels und Vorhersagen zur Auswertung vor
            # Bereite die Labels und Vorhersagen zur Auswertung vor
            true_labels = [[UNIFIED_LABELS[l] for l in label if l != -100] for label in labels]
            pred_labels = [[UNIFIED_LABELS[p] for (p, l) in zip(pred, label) if l != -100] 
                        for pred, label in zip(predictions, labels)]

            # Berechne seqeval-Metriken
            overall_metrics = metric.compute(predictions=pred_labels, references=true_labels, zero_division=1)

            # Flache die Labels und Vorhersagen für die Confusion-Matrix
            true_labels_flat = [label for sublist in true_labels for label in sublist]
            pred_labels_flat = [pred for sublist in pred_labels for pred in sublist]

            # Reduziere die Klassen auf tatsächlich vorkommende Labels
            unique_true_labels = set(true_labels_flat)
            unique_pred_labels = set(pred_labels_flat)
            relevant_classes = sorted(unique_true_labels.union(unique_pred_labels))

            # Berechne die Confusion-Matrix für relevante Klassen
            cm = confusion_matrix(true_labels_flat, pred_labels_flat, labels=relevant_classes)

            # Speichere die Ergebnisse für die Zusammenfassung
            results_summary.append({
                "Adapter": adapter_name,
                "Tested_on": test_dataset_name,
                "accuracy": overall_metrics["overall_accuracy"],
                "f1": overall_metrics["overall_f1"],
                "precision": overall_metrics["overall_precision"],
                "recall": overall_metrics["overall_recall"],
                "confusion_matrix": cm
            })

            # **WICHTIG: Adapter deaktivieren nach dem Test**
            print(f"Deaktiviere Adapter: {adapter_name}")

            # Berechne die prozentuale Confusion-Matrix
            cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100
            annotations = np.array([
                ["{:.2f}%\n({})".format(cm_percentage[i, j], cm[i, j])
                 for j in range(len(relevant_classes))] for i in range(len(relevant_classes))
            ])

            # Visualisiere die Confusion-Matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm_percentage, annot=annotations, fmt='', 
                        xticklabels=relevant_classes, yticklabels=relevant_classes, cmap='Blues')

            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for Adapter {adapter_name} tested on {test_dataset_name}')
            plt.show()

        print(model.adapter_summary())
        model.set_active_adapters(None)  # Deaktiviert den aktiven Adapter
"""
# Ergebnisse zusammenfassen und anzeigen
def summarize_results():
    results_df = pd.DataFrame(results_summary)
    print("\nZusammenfassung der Ergebnisse:")
    print(tabulate(results_df.drop(columns=["confusion_matrix"]), headers="keys", tablefmt="grid", floatfmt=".4f"))

# Beispielaufruf
#evaluate_all_adapters()
summarize_results()

print(model.adapter_summary())

# # 4.) Evaluieren und interpretieren Sie das Netzwerk auf den Test-Daten
"""
def plot_training_progress(trainer):
    log_history = trainer.state.log_history
    train_losses = []
    eval_losses = []
    eval_accuracies = []

    for log in log_history:
        if 'loss' in log:
            train_losses.append(log['loss'])
        if 'eval_loss' in log:
            eval_losses.append(log['eval_loss'])
        if 'eval_accuracy' in log:
            eval_accuracies.append(log['eval_accuracy'])

    if len(eval_losses) == 0 or len(eval_accuracies) == 0:
        print("Warnung: Keine Eval-Daten gefunden. Überprüfe, ob das Evaluationsset korrekt geladen wurde.")
        return

    epochs = range(1, len(eval_losses) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", color='blue')
    plt.plot(epochs, eval_losses, label="Validation Loss", marker="o", color='orange')
    plt.xlabel("Epochen")
    plt.ylabel("Loss")
    plt.title("Train und Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, eval_accuracies, label="Validation Accuracy", marker="o", color='green')
    plt.xlabel("Epochen")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
"""

#adapter_setup = Fuse(Stack(trained_adapters))
#model.delete_adapter_fusion(adapter_setup)

# Optional: Fusion der Adapter
print("Aktiviere Fusion Layer")
adapter_setup = Fuse(*trained_adapters)
model.add_tagging_head("head_fusion", num_labels=len(ID_TO_LABEL), id2label=ID_TO_LABEL)
model.add_adapter_fusion(adapter_setup)

# ✅ Reset W&B environment variables to force a truly separate run
os.environ.pop("WANDB_RUN_ID", None)
os.environ.pop("WANDB_RESUME", None)

# W&B setup
wandb.init(
        project="CrossCorpusNER",   # Dein Projektname in W&B
        name=f"Fusion_{wandb.util.generate_id()}",  # z.B. Laufname = Name des Datensatzes
        tags=[f"timestamp_{START_TIME}", "Fusion"],
    )

lr_fusion = wandb.config.learning_rate if hasattr(wandb.config, "learning_rate") else 2e-4
batch_size_fusion = wandb.config.batch_size if hasattr(wandb.config, "batch_size") else 8
epochs_fusion = wandb.config.num_train_epochs if hasattr(wandb.config, "num_train_epochs") else 1

# Unfreeze and activate fusion setup
model.set_active_adapters(adapter_setup)
model.train_adapter_fusion(adapter_setup)

#model.set_active_adapters(list(adapters.values()))
#model.train_adapter(list(adapters.values()))

# Cross-Domain Trainingsdatensätze für die Fusion vorbereiten
cross_domain_train = concatenate_datasets([tokenized_datasets[ds]["train"] for ds in tokenized_datasets.keys()])
cross_domain_eval = concatenate_datasets([tokenized_datasets[ds]["dev"] for ds in tokenized_datasets.keys()])

fusion_training_args = TrainingArguments(
    output_dir=os.path.join(BASE_ADAPTER_DIR,"results","fusion"),
    evaluation_strategy="epoch",
    learning_rate=lr_fusion,
    per_device_train_batch_size=batch_size_fusion,
    per_device_eval_batch_size=batch_size_fusion,
    num_train_epochs=epochs_fusion,
    weight_decay=0.01,
    logging_dir="./logs/fusion",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
)

fusion_trainer = AdapterTrainer(
    model=model,
    args=fusion_training_args,
    train_dataset=cross_domain_train,
    eval_dataset=cross_domain_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=calculate_metrics,
)

fusion_trainer.train() 
fusion_trainer.evaluate()
print(model.adapter_summary())

fusion_path = os.path.join(BASE_ADAPTER_DIR,"adapters","fusion")
head_path = os.path.join(BASE_ADAPTER_DIR,"heads", "head_fusion")

model.save_adapter_fusion(fusion_path, adapter_setup)
model.save_head(head_path, "head_fusion")
#plot_training_progress(fusion_trainer) 

print("\nEvaluierung des fusionierten Modells...")

for test_dataset_name, datasets in tokenized_datasets.items():
    print(f"Testen auf {test_dataset_name}-Testset...")

    predictions, labels, _ = fusion_trainer.predict(datasets["test"])
    predictions = np.argmax(predictions, axis=-1)

    true_labels = [[UNIFIED_LABELS[l] for l in label if l != -100] for label in labels]
    pred_labels = [[UNIFIED_LABELS[p] for (p, l) in zip(pred, label) if l != -100] 
                   for pred, label in zip(predictions, labels)]

    overall_metrics = metric.compute(predictions=pred_labels, references=true_labels, zero_division=1)

    true_labels_flat = [label for sublist in true_labels for label in sublist]
    pred_labels_flat = [pred for sublist in pred_labels for pred in sublist]

    unique_true_labels = set(true_labels_flat)
    unique_pred_labels = set(pred_labels_flat)
    relevant_classes = sorted(unique_true_labels.union(unique_pred_labels))

    cm = confusion_matrix(true_labels_flat, pred_labels_flat, labels=relevant_classes)

    results_summary.append({
        "Testset": test_dataset_name,
        "accuracy": overall_metrics["overall_accuracy"],
        "f1": overall_metrics["overall_f1"],
        "precision": overall_metrics["overall_precision"],
        "recall": overall_metrics["overall_recall"],
        "confusion_matrix": cm
    })
    """
    wandb.log({
        "test_accuracy": overall_metrics["overall_accuracy"],
        "test_f1": overall_metrics["overall_f1"],
        "test_precision": overall_metrics["overall_precision"],
        "test_recall": overall_metrics["overall_recall"],
    })
    """

    cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100
    annotations = np.array([
        ["{:.2f}%\n({})".format(cm_percentage[i, j], cm[i, j])
         for j in range(len(relevant_classes))] for i in range(len(relevant_classes))
    ])
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percentage, annot=annotations, fmt='', 
                xticklabels=relevant_classes, yticklabels=relevant_classes, cmap='Blues')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for Fusion Model tested on {test_dataset_name}')
    plt.show()
    """
wandb.finish()

results_df = pd.DataFrame(results_summary)
print("\nZusammenfassung der Ergebnisse:")
print(tabulate(results_df.drop(columns=["confusion_matrix"]), headers="keys", tablefmt="grid", floatfmt=".4f"))


