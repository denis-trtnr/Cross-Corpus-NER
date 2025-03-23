# training_evaluation.py

import os
import numpy as np
import pandas as pd
import wandb
import evaluate
from datetime import datetime
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from transformers import TrainingArguments, AutoConfig, AutoTokenizer
from adapters import AutoAdapterModel, AdapterTrainer, AdapterConfig
from adapters.composition import Stack, Fuse

# Importiere zentrale Objekte aus dem Pre‑Processing-Modul.
from data_preprocessing import get_tokenized_datasets, ID_TO_LABEL, data_collator, config, tokenizer, model
# Importiere metrics utils aus dem Metrics-Modul.
from metrics_utils import save_confusion_matrix_png, calculate_metrics, append_average_metrics, summarize_results

metric = evaluate.load("seqeval")

# Laden der tokenisierten Datasets aus dem Pre‑Processing-Modul
tokenized_datasets = get_tokenized_datasets()

BASE_ADAPTER_DIR = "/netscratch/dtrautner/studienarbeit/results"
START_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Globale Variablen zur Speicherung von Adapter-Namen, Trainern und Ergebnissen
trained_adapters = []
adapter_trainers = {}
results_summary = []

def train_with_adapter(dataset_name, tokenized_data):
    """
    Trainiert einen Einzeladapter für einen bestimmten Datensatz.
    Der Adapter wird dem Modell hinzugefügt, aktiviert (hier mittels Stack-Komposition),
    und anschließend wird das Training und die Evaluation (inklusive Confusion Matrix)
    durchgeführt.
    """
    # Reset von W&B-Umgebungsvariablen
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)
    
    wandb.init(
        project="CrossCorpusNER",
        name=f"Train_{dataset_name}_{wandb.util.generate_id()}",
        tags=[f"timestamp_{START_TIME}", dataset_name],
    )
    
    lr = wandb.config.learning_rate if hasattr(wandb.config, "learning_rate") else 2e-4
    batch_size = wandb.config.batch_size if hasattr(wandb.config, "batch_size") else 8
    epochs = wandb.config.num_train_epochs if hasattr(wandb.config, "num_train_epochs") else 1

    adapter_config = AdapterConfig.load("houlsby")
    adapter_name = f"{dataset_name}_adapter"
    print(f"Erstellung des Adapters {adapter_name} ...")
    model.add_adapter(adapter_name, config=adapter_config)
    model.add_tagging_head(adapter_name, num_labels=len(ID_TO_LABEL), id2label=ID_TO_LABEL)
    print("Adapter erstellt.")
    
    # Aktiviere den Adapter über eine Stack-Komposition
    model.set_active_adapters(Stack(adapter_name))
    model.train_adapter(adapter_name)

    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_ADAPTER_DIR, "results", dataset_name),
        eval_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=f"./logs/{dataset_name}",
        logging_strategy="epoch",
        remove_unused_columns=False,
        report_to="wandb",
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=calculate_metrics,
    )

    trainer.train()
    trainer.evaluate()

    adapter_path = os.path.join(BASE_ADAPTER_DIR, "adapters", adapter_name)
    head_path = os.path.join(BASE_ADAPTER_DIR, "heads", adapter_name)
    model.save_adapter(adapter_path, adapter_name)
    model.save_head(head_path, adapter_name)
    
    trained_adapters.append(adapter_name)
    adapter_trainers[adapter_name] = trainer
    print(model.adapter_summary())

    # Evaluation auf allen Test-Sets
    for test_dataset_name, dataset in tokenized_datasets.items():
        print(f"Evaluierung auf {test_dataset_name}-Testset ...")
        predictions, labels, _ = trainer.predict(dataset["test"])
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

        # Speichere Confusion Matrix
        save_confusion_matrix_png(cm, relevant_classes, f"confusion_matrix_{adapter_name}_{test_dataset_name}.png",
                          title=f"Confusion Matrix for {adapter_name} tested on {test_dataset_name}")

        # Speichere die Ergebnisse für die Zusammenfassung
        results_summary.append({
            "Adapter": adapter_name,
            "Testset": test_dataset_name,
            "accuracy": overall_metrics["overall_accuracy"],
            "f1": overall_metrics["overall_f1"],
            "precision": overall_metrics["overall_precision"],
            "recall": overall_metrics["overall_recall"],
            "confusion_matrix": cm
        })
    
    wandb.finish()
    model.set_active_adapters(None)

def train_all_adapters():
    """Trainiert für alle in tokenized_datasets vorhandenen Datensätze einen Adapter."""
    for dataset_name, tokenized_data in tokenized_datasets.items():
        print(f"Starte Finetuning für {dataset_name}...")
        train_with_adapter(dataset_name, tokenized_data)
    adapter_summary_file_name= f"adapter_summary_{START_TIME}.csv"
    summarize_results(results_summary, adapter_summary_file_name)
    print(model.adapter_summary())

# Optional: Training des Fusion-Layers, der die trainierten Einzeladapter kombiniert.
def train_fusion_layer():
    """
    Kombiniert die trainierten Einzeladapter mittels Fusion.
    Dabei werden nur die Fusionsparameter trainiert, während die Einzeladapter fix bleiben.
    """
    print("\nTrainiere Fusion Layer:")
    adapter_setup = Fuse(*trained_adapters)
    model.add_tagging_head("head_fusion", num_labels=len(ID_TO_LABEL), id2label=ID_TO_LABEL)
    model.add_adapter_fusion(adapter_setup)

    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)
    wandb.init(
        project="CrossCorpusNER",
        name=f"Fusion_{wandb.util.generate_id()}",
        tags=[f"timestamp_{START_TIME}", "Fusion"],
    )
    lr_fusion = wandb.config.learning_rate if hasattr(wandb.config, "learning_rate") else 2e-4
    batch_size_fusion = wandb.config.batch_size if hasattr(wandb.config, "batch_size") else 8
    epochs_fusion = wandb.config.num_train_epochs if hasattr(wandb.config, "num_train_epochs") else 1

    model.set_active_adapters(adapter_setup)
    model.train_adapter_fusion(adapter_setup)

    # Cross-Domain-Daten: Zusammenführen aller Train- und Dev-Sets
    from datasets import concatenate_datasets
    cross_domain_train = concatenate_datasets([tokenized_datasets[ds]["train"] for ds in tokenized_datasets])
    cross_domain_eval = concatenate_datasets([tokenized_datasets[ds]["dev"] for ds in tokenized_datasets])
    
    fusion_training_args = TrainingArguments(
        output_dir=os.path.join(BASE_ADAPTER_DIR, "results", "fusion"),
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

    fusion_path = os.path.join(BASE_ADAPTER_DIR, "adapters", "fusion")
    head_path = os.path.join(BASE_ADAPTER_DIR, "heads", "head_fusion")
    model.save_adapter_fusion(fusion_path, adapter_setup)
    model.save_head(head_path, "head_fusion")

    fusion_summary = []

    # Evaluation des fusionierten Modells
    for test_dataset_name, dataset in tokenized_datasets.items():
        print(f"Testen auf {test_dataset_name}-Testset mit Fusion-Modell ...")
        predictions, labels, _ = fusion_trainer.predict(dataset["test"])
        predictions = np.argmax(predictions, axis=-1)
        true_labels = [[ID_TO_LABEL[l] for l in label if l != -100] for label in labels]
        pred_labels = [
            [ID_TO_LABEL[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        overall_metrics = metric.compute(predictions=pred_labels, references=true_labels, zero_division=1)

        true_labels_flat = [label for sublist in true_labels for label in sublist]
        pred_labels_flat = [pred for sublist in pred_labels for pred in sublist]

        unique_true_labels = set(true_labels_flat)
        unique_pred_labels = set(pred_labels_flat)
        relevant_classes = sorted(unique_true_labels.union(unique_pred_labels))

        cm = confusion_matrix(true_labels_flat, pred_labels_flat, labels=relevant_classes)

        # Speichere Confusion Matrix
        save_confusion_matrix_png(cm, relevant_classes, f"confusion_matrix_fusion_{test_dataset_name}.png",
                          title=f"Confusion Matrix for Fusion Adapter tested on {test_dataset_name}")

        fusion_summary.append({
            "Testset": test_dataset_name,
            "accuracy": overall_metrics["overall_accuracy"],
            "f1": overall_metrics["overall_f1"],
            "precision": overall_metrics["overall_precision"],
            "recall": overall_metrics["overall_recall"],
            "confusion_matrix": cm
        })
    wandb.finish()

    fusion_summary_file_name= f"fusion_summary_{START_TIME}.csv"
    summarize_results(fusion_summary, fusion_summary_file_name)
    print(model.adapter_summary())

if __name__ == "__main__":
    # Einzeladapter trainieren und evaluieren
    train_all_adapters()

    # Optional: Fusion der trainierten Einzeladapter
    train_fusion_layer()
