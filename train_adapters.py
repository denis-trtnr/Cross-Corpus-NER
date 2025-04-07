import os
import numpy as np
import pandas as pd
import wandb
import evaluate
from datasets import concatenate_datasets
from datetime import datetime
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from transformers import TrainingArguments, AutoConfig, AutoTokenizer
from adapters import AutoAdapterModel, AdapterTrainer, AdapterConfig, SeqBnConfig, CompacterConfig
from adapters.composition import Stack, Fuse

# Importiere den DataPreprocessoraus dem Pre‑Processing-Modul.
from data_preprocessing import DataPreprocessor
# Importiere metrics utils aus dem Metrics-Modul.
from metrics_utils import save_confusion_matrix_png, calculate_metrics, append_average_metrics, summarize_results
from config_utils import load_config, read_yaml_config

global_config = read_yaml_config()

# Werte aus der Konfiguration extrahieren
base_model_name = global_config.get(
    "model", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)
mapping_type = global_config.get("mapping_type", "granular")
data_dir = global_config.get("data_dir", "data")

preprocessor = DataPreprocessor(
    mapping_type=mapping_type,
    base_model=base_model_name,
    data_dir=data_dir,
)

#Initialisiere globale Variablen
tokenized_datasets = preprocessor.get_tokenized_datasets()
tokenizer = preprocessor.tokenizer
model = preprocessor.model
data_collator = preprocessor.data_collator
ID_TO_LABEL = preprocessor.id_to_label

metric = evaluate.load("seqeval")

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
    
    unique_run_id = wandb.util.generate_id()
    run = wandb.init(
        project="Cross-Corpus-NER",
        name=f"Train_{dataset_name}_{unique_run_id}",
        tags=[f"timestamp_{START_TIME}", dataset_name],
        reinit=True,
        resume=False,
        id=unique_run_id,
    )

    lr = getattr(wandb.config, "learning_rate", None) or getattr(global_config, "learning_rate", 2e-4)
    batch_size = getattr(wandb.config, "batch_size", None) or getattr(global_config, "batch_size", 8)
    epochs = getattr(wandb.config, "num_train_epochs", None) or getattr(global_config, "num_train_epochs", 1)
    adapter_config_name = getattr(wandb.config, "adapter_config_name", None) or getattr(global_config, "adapter_config_name", "houlsby")


    # Adapterkonfiguration basierend auf Name
    adapter_config = None

    if adapter_config_name.lower() == "custom":
        print("Custom Adapter-Konfiguration mit Default-Parametern wird erstellt...")
        adapter_config = SeqBnConfig(
            mh_adapter=True,
            output_adapter=True,
            reduction_factor=16,
            non_linearity="gelu",
            ln_before=False,
            ln_after=True,
            residual_before_ln=True
        )
    elif adapter_config_name.lower() == "compacter":
        print("Compacter Adapter-Konfiguration wird erstellt...")
        adapter_config = CompacterConfig(
            reduction_factor=16,
            non_linearity="gelu"
        )
    else:
        print(f"Adapter-Konfiguration '{adapter_config_name}' wird geladen...")
        adapter_config = AdapterConfig.load(adapter_config_name)

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
        local_rank=-1,
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["dev"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda pred: calculate_metrics(pred, ID_TO_LABEL)
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
        save_confusion_matrix_png(cm, relevant_classes, f"confusion_matrix_{adapter_name}_{test_dataset_name}_{base_model_name}_{adapter_config_name}_{mapping_type}.png",
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
    
    run.finish()
    model.set_active_adapters(None)

def train_all_adapters():
    """Trainiert für alle in tokenized_datasets vorhandenen Datensätze einen Adapter."""
    for dataset_name, tokenized_data in tokenized_datasets.items():
        print(f"Starte Finetuning für {dataset_name}...")
        train_with_adapter(dataset_name, tokenized_data)
    adapter_summary_file_name= f"adapter_summary_{base_model_name}_{START_TIME}_{mapping_type}.csv"
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

    unique_run_id = wandb.util.generate_id()
    run = wandb.init(
        project="Cross-Corpus-NER",
        name=f"Fusion_{unique_run_id}",
        tags=[f"timestamp_{START_TIME}", "Fusion"],
        reinit=True,
        resume=False,
        id=unique_run_id,
    )

    lr_fusion = getattr(wandb.config, "learning_rate", None) or getattr(global_config, "learning_rate", 2e-4)
    batch_size_fusion = getattr(wandb.config, "batch_size", None) or getattr(global_config, "batch_size", 8)
    epochs_fusion = getattr(wandb.config, "num_train_epochs_fusion", None) or getattr(global_config, "num_train_epochs_fusion", 1)
    adapter_config_name = getattr(wandb.config, "adapter_config_name", None) or getattr(global_config, "adapter_config_name", "houlsby")

    model.set_active_adapters(adapter_setup)
    model.train_adapter_fusion(adapter_setup)

    # Cross-Domain-Daten: Zusammenführen aller Train- und Dev-Sets
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
        local_rank=-1,
    )

    fusion_trainer = AdapterTrainer(
        model=model,
        args=fusion_training_args,
        train_dataset=cross_domain_train,
        eval_dataset=cross_domain_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda pred: calculate_metrics(pred, ID_TO_LABEL)
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
        save_confusion_matrix_png(cm, relevant_classes, f"confusion_matrix_fusion_{test_dataset_name}_{base_model_name}_{adapter_config_name}_{mapping_type}.png",
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

    fusion_summary_file_name= f"fusion_summary_{base_model_name}_{START_TIME}_{adapter_config_name}_{mapping_type}.csv"
    summarize_results(fusion_summary, fusion_summary_file_name)
    print(model.adapter_summary())

if __name__ == "__main__":
    # Einzeladapter trainieren und evaluieren
    train_all_adapters()

    # Optional: Fusion der trainierten Einzeladapter
    train_fusion_layer()
