import os
import evaluate
from datetime import datetime
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification

# Importiere den DataPreprocessoraus dem Pre‑Processing-Modul.
from data_preprocessing import DataPreprocessor
# Importiere metrics utils aus dem Metrics-Modul.
from metrics_utils import calculate_metrics, summarize_results, evaluate_model_on_testsets
from config_utils import read_yaml_config


class BaselineTrainerManager:
    """
    Verwaltet das Training klassischer Transformer-Modelle (ohne Adapter) 
    auf Basis eines vorverarbeiteten Datasets.
    
    Alle für das Training notwendigen Komponenten (Modell, Tokenizer, Datasets etc.)
    müssen beim Initialisieren übergeben werden
    """ 
    def __init__(
        self,
        base_model_name,
        mapping_type,
        tokenized_datasets,
        tokenizer,
        data_collator,
        id_to_label,
        model_config,
        config=None,
        base_adapter_dir="/netscratch/dtrautner/studienarbeit/results",
    ):
        """
        Initialisiert das Baseline-Training mit vorbereiteten Komponenten.

        Parameter:
        - base_model_name (str): Name des Basismodells
        - mapping_type (str): Mapping-Strategie (z.B. granular, broad)
        - tokenized_datasets (dict): Tokenisierte Datensätze für Training, Validierung und Test
        - tokenizer: Der verwendete Tokenizer
        - data_collator: Der Data Collator für die Dataloader
        - id_to_label (dict): Mapping von IDs zu Labels
        - model_config (dict): Konfiguration zur Initialisierung des Modells
        - config (dict): Optional geladene YAML-Konfiguration
        - base_adapter_dir (str): Zielverzeichnis für Ergebnisse und Logs
        """
        self.config = config or read_yaml_config()
        self.base_model_name = base_model_name
        self.mapping_type = mapping_type
        self.base_adapter_dir = base_adapter_dir
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.id_to_label = id_to_label
        self.label_to_id = {v: k for k, v in id_to_label.items()}

        self.model = AutoModelForTokenClassification.from_pretrained(
                base_model_name,
                config=model_config
        )
        self.tokenized_datasets = tokenized_datasets
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        

        self.metric = evaluate.load("seqeval")
        self.results_summary = []


    def train_all_models(self):
        """
        Trainiert ein klassisches Transformer-Modell (ohne Adapter)
        separat auf jedem Datensatz in den übergebenen tokenisierten Datasets.
        """
        for dataset_name, tokenized_data in self.tokenized_datasets.items():
            print(f"Starte Modell-Finetuning für {dataset_name}...")

            training_args = TrainingArguments(
                output_dir=os.path.join(self.base_adapter_dir, "results", dataset_name),
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                num_train_epochs=5,
                per_device_train_batch_size=8,
                weight_decay=0.01,
                load_best_model_at_end=True,
                logging_dir=f"./logs/{dataset_name}",
                logging_strategy="epoch",
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_data["train"],
                eval_dataset=tokenized_data["dev"],
                processing_class=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=lambda pred: calculate_metrics(pred, self.id_to_label),
            )

            trainer.train()
            trainer.evaluate()

            eval_results = evaluate_model_on_testsets(
                trainer=trainer,
                adapter_name=f"{dataset_name}_model",
                id_to_label=self.id_to_label,
                tokenized_datasets=self.tokenized_datasets,
                base_model_name=self.base_model_name,
                adapter_config_name="N/A",
                mapping_type=self.mapping_type,
            )
            self.results_summary.extend(eval_results)

        # Speichern der Ergebnisse
        file_name = f"baseline_summary_{self.base_model_name}_{self.start_time}_{self.mapping_type}.csv"
        summarize_results(self.results_summary, file_name)



if __name__ == "__main__":
    print("🚀 Starte BaselineTrainerManager...")

    # Konfiguration laden
    global_config = read_yaml_config()

    # Werte aus der Konfiguration extrahieren
    base_model_name = global_config.get(
        "model", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )
    mapping_type = global_config.get("mapping_type", "granular")

    # Erstelle Preprocessor
    pre = DataPreprocessor(
        mapping_type=mapping_type,
        base_model=base_model_name,
    )

    # Erstelle eine Instanz des AdapterTrainerManagers
    manager = BaselineTrainerManager(
        base_model_name=base_model_name,
        mapping_type=mapping_type,
        config=global_config,
        tokenized_datasets=pre.get_tokenized_datasets(),
        tokenizer=pre.tokenizer,
        data_collator=pre.data_collator,
        id_to_label=pre.id_to_label,
    )

    # Starte Training
    manager.train_all_models()