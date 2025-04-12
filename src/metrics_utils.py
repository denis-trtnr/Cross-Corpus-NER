import os
import numpy as np
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
VISUALIZATION_DIR = os.path.join(BASE_DIR, "..", "visualizations")

# Load the seqeval metric once
metric = evaluate.load("seqeval")

def calculate_metrics(predictions_and_labels, id_to_label):
    """
    Berechnet NER-Metriken (Accuracy, F1, Precision, Recall) mithilfe von seqeval.
    
    Parameter:
      predictions_and_labels (tuple): Ein Tupel (logits, labels), wobei logits ein 
        Array der Modell-Ausgaben und labels die wahren Label sind.
        
    Rückgabe:
      dict: Ein Dictionary mit den berechneten Metriken.
    """
    logits, labels = predictions_and_labels
    predicted_labels = np.argmax(logits, axis=-1)

    true_labels = []
    true_predictions = []
    for sentence_labels, sentence_predictions in zip(labels, predicted_labels):
        filtered_true = []
        filtered_pred = []
        for lab, pred in zip(sentence_labels, sentence_predictions):
            if lab != -100:
                filtered_true.append(id_to_label[lab])
                filtered_pred.append(id_to_label[pred])
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



def evaluate_model_on_testsets(trainer, adapter_name, id_to_label, tokenized_datasets, base_model_name, adapter_config_name, mapping_type, prefix=""):
    """
    Führt die Evaluation eines trainierten Adapters oder Fusion-Modells auf allen Test-Datensätzen durch.
    Dabei werden Vorhersagen generiert, seqeval-Metriken berechnet und Confusion-Matrizen erstellt und gespeichert.

    Parameter:
      trainer (AdapterTrainer): Der Trainer, mit dem Vorhersagen gemacht werden.
      adapter_name (str): Name des Adapters oder Fusions-Setups.
      id_to_label (dict): Mapping von Label-IDs zu Label-Namen.
      tokenized_datasets (dict): Tokenisierte Datensätze mit "test"-Splits für jede Domäne.
      base_model_name (str): Name des Basis-Modells (für die Datei- und Logbenennung).
      adapter_config_name (str): Name der Adapter-Konfiguration.
      mapping_type (str): Typ der verwendeten Label-Mapping-Konfiguration (z.B. granular).
      prefix (str): Optionaler Präfix für Dateinamen (z.B. "fusion_").

    Rückgabe:
      list: Eine Liste von Dictionaries, die die Metriken und Confusion-Matrizen pro Testset enthalten.
    """
    
    evaluation_results = []

    for test_dataset_name, dataset in tokenized_datasets.items():
        print(f"Evaluierung auf {test_dataset_name}-Testset ...")
        predictions, labels, _ = trainer.predict(dataset["test"])
        predictions = np.argmax(predictions, axis=-1)
        true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
        pred_labels = [
            [id_to_label[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        overall_metrics = metric.compute(predictions=pred_labels, references=true_labels, zero_division=1)

        true_labels_flat = [label for sublist in true_labels for label in sublist]
        pred_labels_flat = [pred for sublist in pred_labels for pred in sublist]

        relevant_classes = sorted(set(true_labels_flat).union(set(pred_labels_flat)))
        cm = confusion_matrix(true_labels_flat, pred_labels_flat, labels=relevant_classes)

        filename = f"confusion_matrix_{prefix}{adapter_name}_{test_dataset_name}_{base_model_name}_{adapter_config_name}_{mapping_type}.png"
        save_confusion_matrix_png(cm, relevant_classes, filename, title=f"Confusion Matrix for {adapter_name} tested on {test_dataset_name}")

        evaluation_results.append({
            "Adapter": adapter_name if prefix != "fusion_" else "Fusion",
            "Testset": test_dataset_name,
            "accuracy": overall_metrics["overall_accuracy"],
            "f1": overall_metrics["overall_f1"],
            "precision": overall_metrics["overall_precision"],
            "recall": overall_metrics["overall_recall"],
            "confusion_matrix": cm
        })
    
    return evaluation_results


def append_average_metrics(results_summary):
    """
    Berechnet den arithmetischen Durchschnitt der Zusammengefassten Metriken
    """
    n = len(results_summary)
    if n == 0:
        return results_summary

    avg_accuracy = sum(entry["accuracy"] for entry in results_summary) / n
    avg_f1       = sum(entry["f1"] for entry in results_summary) / n
    avg_precision = sum(entry["precision"] for entry in results_summary) / n
    avg_recall    = sum(entry["recall"] for entry in results_summary) / n

    avg_entry = {
          "Testset": "Durchschnitt",
          "accuracy": avg_accuracy,
          "f1": avg_f1,
          "precision": avg_precision,
          "recall": avg_recall,
          "confusion_matrix": None  # Aggregierte Confusion Matrix wird hier nicht berechnet
    }

    results_summary.append(avg_entry)
    return results_summary

def summarize_results(results_summary, filename, output_folder=RESULTS_DIR):
    """Fasst die Evaluationsergebnisse zusammen und gibt eine Tabelle aus."""
    df = pd.DataFrame(append_average_metrics(results_summary))
    print("\nZusammenfassung der Ergebnisse:")
    print(tabulate(df.drop(columns=["confusion_matrix"]), headers="keys", tablefmt="grid", floatfmt=".4f"))

    # Ordner erstellen, falls er nicht existiert
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Pfad zur Ausgabedatei
    output_path = os.path.join(output_folder, filename.replace("/", "-"))

    # Speichere das DataFrame als CSV
    df.drop(columns=["confusion_matrix"]).to_csv(output_path, index=False)
    print(f"\nErgebnisse wurden in '{output_path}' gespeichert.")


def save_confusion_matrix_png(cm, relevant_classes, filename, title="Confusion Matrix", folder=VISUALIZATION_DIR):
    """
    Erstellt eine Heatmap aus einer Konfusionsmatrix und speichert diese als PNG-Datei.
    
    Parameter:
      cm (np.array): Die Konfusionsmatrix.
      relevant_classes (list): Eine sortierte Liste der Klassennamen (für x- und y-Achse).
      filename (str): Der Dateiname (inklusive Pfad), unter dem das PNG gespeichert wird.
      title (str): Der Titel der Heatmap.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, filename.replace("/", "-"))

    # Vermeide Division durch 0: Falls eine Zeilensumme 0 ist, setze sie auf 1
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Dadurch wird für leere Zeilen der Prozentsatz 0
    
    # Erstelle relative Werte für Confusion matrix
    cm_percentage = cm / row_sums * 100
    annotations = np.array([
        ["{:.2f}%\n({})".format(cm_percentage[i, j], cm[i, j])
         for j in range(len(relevant_classes))]
        for i in range(len(relevant_classes))
    ])

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_percentage,                # Daten für die Farbskala
        annot=annotations,           # Annotationen mit Prozent + Absolutwerten
        fmt="",                       # Kein eigenes Zahlenformat, da annotations schon formatiert
        cmap="Blues", 
        xticklabels=relevant_classes,
        yticklabels=relevant_classes,
        vmin=0, vmax=100             # Farbspektrum von 0% bis 100%
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()