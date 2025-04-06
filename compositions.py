# compositions.py

import os
import evaluate
import numpy as np
import pandas as pd
import adapters.composition as ac
from datetime import datetime
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from transformers import TrainingArguments
from adapters import AdapterTrainer

# Importiere den DataPreprocessor
from data_preprocessing import DataPreprocessor
# Importiere metrics utils aus dem Metrics-Modul.
from metrics_utils import save_confusion_matrix_png, calculate_metrics, append_average_metrics, summarize_results

#Variables zu setzen
mapping_type = "granular"
base_model = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


# Globale Variablen
BASE_ADAPTER_DIR = "/netscratch/dtrautner/studienarbeit/results"
START_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
metric = evaluate.load("seqeval")

preprocessor = DataPreprocessor(
    mapping_type=mapping_type,
    base_model=base_model,
    data_dir="data"
)

tokenized_datasets = preprocessor.get_tokenized_datasets()
tokenizer = preprocessor.tokenizer
model = preprocessor.model
data_collator = preprocessor.data_collator
ID_TO_LABEL = preprocessor.id_to_label

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
        
        # Initialisiere einen AdapterTrainer ausschließlich für die Vorhersage
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=8,
            evaluation_strategy="no"  # Es findet kein Training statt
        )

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator
        )


        # Vorhersagen generieren
        predictions_output = trainer.predict(dataset["test"])

        # Extrahiere Logits und Label-IDs aus dem predictions_output-Objekt
        logits = predictions_output.predictions
        labels = predictions_output.label_ids

        # Jetzt kann np.argmax angewendet werden
        predictions = np.argmax(logits, axis=-1)
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
    compositions_summary_file_name= f"{method_name}_summary_{base_model}_{START_TIME}.csv"
    summarize_results(results, compositions_summary_file_name)
    return results


def test_all_compositions():
    """
    Testet alle Kompositionsmethoden (außer Fuse) und gibt eine Übersicht der Ergebnisse aus.
    """
    # Definiere die Kompositionsmethoden, die getestet werden sollen.
    # Hier: "stack", "average" und "parallel".
    compositions = {
        "stack": ac.Stack(*adapter_names),
        "average": ac.Average(*adapter_names, weights=[0.2, 0.2, 0.2, 0.2, 0.2])
    }
    
    all_results = []
    
    for method, comp_obj in compositions.items():
        print(f"\nTeste Komposition: {method}")
        results = evaluate_composition(method, comp_obj)
        all_results.extend(results)


    compositions_summary_file_name= f"compositions_summary_{base_model}_{START_TIME}.csv"
    summarize_results(all_results, compositions_summary_file_name)

    # Deaktiviere die aktiven Adapter nach der Evaluation
    model.set_active_adapters(None)

if __name__ == "__main__":
    test_all_compositions()
