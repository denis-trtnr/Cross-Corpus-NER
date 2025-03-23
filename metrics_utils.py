import os
import numpy as np
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Importiere zentrale Objekte aus dem Pre‑Processing-Modul.
from data_preprocessing import ID_TO_LABEL


# Load the seqeval metric once
metric = evaluate.load("seqeval")

def calculate_metrics(predictions_and_labels):
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
        # Here, we assume that sentence_labels and sentence_predictions are iterables.
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

def summarize_results(results_summary):
    """Fasst die Evaluationsergebnisse zusammen und gibt eine Tabelle aus."""
    df = pd.DataFrame(append_average_metrics(results_summary))
    print("\nZusammenfassung der Ergebnisse:")
    print(tabulate(df.drop(columns=["confusion_matrix"]), headers="keys", tablefmt="grid", floatfmt=".4f"))


def save_confusion_matrix_png(cm, relevant_classes, filename, title="Confusion Matrix", folder="visualizations"):
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

    file_path = os.path.join(folder, filename)

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