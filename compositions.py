# test_compositions.py

import os
import numpy as np
import pandas as pd
import adapters.composition as ac
from datetime import datetime
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
import evaluate

# Importiere zentrale Objekte aus deinem Pre‑Processing‑Modul.
from data_preprocessing import (
    get_tokenized_datasets,
    ID_TO_LABEL,
    data_collator,
    config,
    tokenizer,
    model  # Das Modell, das im Pre‑Processing bereits geladen wurde
)

# Globale Variablen
BASE_ADAPTER_DIR = "/netscratch/dtrautner/studienarbeit/results"
START_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
metric = evaluate.load("seqeval")

# Liste der Namen der bereits trainierten Einzeladapter (angepasst an deine Adapternamen)
adapter_names = [
    "SETH_adapter",
    "Variome_adapter",
    "Variome120_adapter",
    "Amia_adapter",
    "TmVar_adapter"
]

# Funktion, um gespeicherte Adapter aus dem Dateisystem zu laden.
def load_stored_adapters(adapter_names):
    """
    Lädt die gespeicherten Adapter und zugehörigen Heads in das Modell.
    """
    for name in adapter_names:
        adapter_path = os.path.join(BASE_ADAPTER_DIR, "adapters", name)
        head_path = os.path.join(BASE_ADAPTER_DIR, "heads", name)
        print(f"Lade Adapter {name} aus {adapter_path} ...")
        model.load_adapter(adapter_path, load_as=name)
        print(f"Lade Head {name} aus {head_path} ...")
        model.load_head(head_path, load_as=name)
    return adapter_names

# Lade die Adapter ins Modell
load_stored_adapters(adapter_names)

# Hole die tokenisierten Datasets aus dem Pre‑Processing-Modul
tokenized_datasets = get_tokenized_datasets()

# Definition einer calculate_metrics-Funktion (falls sie nicht bereits in einem anderen Modul ist)
def calculate_metrics(predictions_and_labels):
    logits, labels = predictions_and_labels
    predicted_labels = np.argmax(logits, axis=-1)
    true_labels = []
    true_predictions = []
    for sentence_labels, sentence_predictions in zip(labels, predicted_labels):
        # Sicherstellen, dass wir mit iterablen Sätzen arbeiten:
        if np.isscalar(sentence_labels):
            sentence_labels = [sentence_labels]
        if np.isscalar(sentence_predictions):
            sentence_predictions = [sentence_predictions]
        filtered_true = []
        filtered_pred = []
        for lab, pred in zip(sentence_labels, sentence_predictions):
            if lab != -100:
                filtered_true.append(ID_TO_LABEL[lab])
                filtered_pred.append(ID_TO_LABEL[pred])
        if filtered_true:
            true_labels.append(filtered_true)
            true_predictions.append(filtered_pred)
    overall = metric.compute(predictions=true_predictions, references=true_labels, zero_division=1)
    return {
        "accuracy": overall["overall_accuracy"],
        "f1": overall["overall_f1"],
        "precision": overall["overall_precision"],
        "recall": overall["overall_recall"],
    }

def evaluate_composition(method_name, composition_obj):
    """
    Setzt die aktive Komposition im Modell, evaluiert auf allen Testsets und
    gibt eine Ergebnisübersicht zurück.
    """
    # Setze die aktive Adapter-Komposition im Modell
    model.set_active_adapters(composition_obj)
    
    results = []
    # Iteriere über alle Testsets in tokenized_datasets
    for test_dataset_name, dataset in tokenized_datasets.items():
        print(f"Evaluierung mit {method_name} auf {test_dataset_name}-Testset...")
        # Nutze den integrierten Trainer für die Vorhersage
        # (Da hier kein Trainer existiert, simulieren wir die Vorhersage mit model.predict())
        # Du kannst alternativ einen einfachen Trainer initialisieren, falls erforderlich.
        predictions, labels, _ = model.predict(dataset["test"])
        predictions = np.argmax(predictions, axis=-1)
        true_labels = [[ID_TO_LABEL[l] for l in label if l != -100] for label in labels]
        pred_labels = [
            [ID_TO_LABEL[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
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
        results.append({
            "Composition": method_name,
            "Testset": test_dataset_name,
            "accuracy": overall_metrics["overall_accuracy"],
            "f1": overall_metrics["overall_f1"],
            "precision": overall_metrics["overall_precision"],
            "recall": overall_metrics["overall_recall"],
            "confusion_matrix": cm
        })
    return results


def test_all_compositions():
    """
    Testet alle Kompositionsmethoden (außer Fuse) und gibt eine Übersicht der Ergebnisse aus.
    """
    # Definiere die Kompositionsmethoden, die getestet werden sollen.
    # Hier: "stack", "average" und "parallel".
    compositions = {
        "stack": ac.Stack(*adapter_names),
        "average": ac.AverageAdapter(*adapter_names),
        "parallel": ac.ParallelAdapter(*adapter_names)
    }
    
    all_results = []
    
    for method, comp_obj in compositions.items():
        print(f"\nTeste Komposition: {method}")
        results = evaluate_composition(method, comp_obj)
        all_results.extend(results)


    compositions_summary_file_name= f"compositions_summary_{START_TIME}.csv"
    summarize_results(all_results, compositions_summary_file_name)

    # Deaktiviere die aktiven Adapter nach der Evaluation
    model.set_active_adapters(None)

if __name__ == "__main__":
    test_all_compositions()
