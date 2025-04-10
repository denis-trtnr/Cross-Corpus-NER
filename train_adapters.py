import os
import wandb
import evaluate
from datasets import concatenate_datasets
from datetime import datetime
from transformers import TrainingArguments, AutoConfig, AutoTokenizer
from adapters import AutoAdapterModel, AdapterTrainer, AdapterConfig, SeqBnConfig, CompacterConfig
from adapters.composition import Stack, Fuse

# Importiere den DataPreprocessoraus dem Pre‑Processing-Modul.
from data_preprocessing import DataPreprocessor
# Importiere metrics utils aus dem Metrics-Modul.
from metrics_utils import calculate_metrics, summarize_results, evaluate_model_on_testsets
from config_utils import read_yaml_config


class AdapterTrainerManager:
    """
    Verwaltet das Training von Adaptern (einzeln und fusioniert) auf Basis eines vorverarbeiteten Datasets.
    
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
        config=None,
        base_adapter_dir="/netscratch/dtrautner/studienarbeit/results",
    ):
        """
        Initialisiert das Adapter-Training mit vorbereiteten Komponenten.

        Parameter:
        - base_model_name (str): Name des Basismodells
        - mapping_type (str): Mapping-Strategie (z.B. granular, broad)
        - tokenized_datasets (dict): Tokenisierte Datensätze für Training, Validierung und Test
        - tokenizer: Der verwendete Tokenizer
        - data_collator: Der Data Collator für die Dataloader
        - id_to_label (dict): Mapping von IDs zu Labels
        - config (dict): Optional geladene YAML-Konfiguration
        - base_adapter_dir (str): Basisverzeichnis zum Speichern von Adaptern, Logs etc.
        """
        self.config = config or read_yaml_config()
        self.base_model_name = base_model_name
        self.mapping_type = mapping_type
        self.base_adapter_dir = base_adapter_dir
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.model = AutoAdapterModel.from_pretrained(base_model_name)
        self.tokenized_datasets = tokenized_datasets
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.id_to_label = id_to_label

        self.metric = evaluate.load("seqeval")
        self.trained_adapters = []
        self.adapter_trainers = {}
        self.results_summary = []

    def _build_adapter_config(self, config_name):
        """
        Erstellt eine AdapterConfig basierend auf dem Konfigurationsnamen.
        Unterstützt 'custom', 'compacter' oder einen geladenen Adapter-Typ.
        """
        if config_name.lower() == "custom":
            return SeqBnConfig(
                mh_adapter=getattr(wandb.config, "mh_adapter", True),
                output_adapter=True,
                reduction_factor=getattr(wandb.config, "reduction_factor", 8),
                non_linearity=getattr(wandb.config, "non_linearity", "gelu"),
                ln_before=False,
                ln_after=False,
                residual_before_ln=getattr(wandb.config, "residual_before_ln", True),
                dropout=getattr(wandb.config, "dropout", 0.1),
                init_weights=getattr(wandb.config, "init_weights", "bert"),
            )
        elif config_name.lower() == "compacter":
            return CompacterConfig(reduction_factor=16, non_linearity="gelu")
        else:
            return AdapterConfig.load(config_name)

    def train_all_adapters(self):
        """
        Führt das Training für alle verfügbaren Einzel-Datensätze durch.
        Speichert die Adapter, ihre Heads und eine Übersicht der Ergebnisse.
        """
        config_name = None
        for dataset_name, tokenized_data in self.tokenized_datasets.items():
            print(f"Starte Finetuning für {dataset_name}...")

            # Reset von W&B-Umgebungsvariablen
            os.environ.pop("WANDB_RUN_ID", None)
            os.environ.pop("WANDB_RESUME", None)

            run_id = wandb.util.generate_id()
            run = wandb.init(
                project="Cross-Corpus-NER",
                name=f"Train_{dataset_name}_{run_id}",
                tags=[f"timestamp_{self.start_time}", dataset_name],
                reinit=True,
                resume=False,
                id=run_id,
            )

            config_name = getattr(wandb.config, "adapter_config_name", None) or self.config.get("adapter_config_name", "houlsby")
            adapter_config = self._build_adapter_config(config_name)

            adapter_name = f"{dataset_name}_adapter"
            self.model.add_adapter(adapter_name, config=adapter_config)
            self.model.add_tagging_head(adapter_name, num_labels=len(self.id_to_label), id2label=self.id_to_label)
            self.model.set_active_adapters(Stack(adapter_name))
            self.model.train_adapter(adapter_name)

            lr = getattr(wandb.config, "learning_rate", None) or self.config.get("learning_rate", 2e-4)
            batch_size = getattr(wandb.config, "batch_size", None) or self.config.get("batch_size", 8)
            epochs = getattr(wandb.config, "num_train_epochs", None) or self.config.get("num_train_epochs", 1)

            training_args = TrainingArguments(
                output_dir=os.path.join(self.base_adapter_dir, "results", dataset_name),
                evaluation_strategy="epoch",
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

            # Speichern des Adapters und des Heads
            self.model.save_adapter(os.path.join(self.base_adapter_dir, "adapters", adapter_name), adapter_name)
            self.model.save_head(os.path.join(self.base_adapter_dir, "heads", adapter_name), adapter_name)

            self.trained_adapters.append(adapter_name)
            self.adapter_trainers[adapter_name] = trainer

            eval_results = evaluate_model_on_testsets(
                trainer=trainer,
                adapter_name=adapter_name,
                id_to_label=self.id_to_label,
                tokenized_datasets=self.tokenized_datasets,
                base_model_name=self.base_model_name,
                adapter_config_name=config_name,
                mapping_type=self.mapping_type,
            )
            self.results_summary.extend(eval_results)
            run.finish()
            self.model.set_active_adapters(None)

        # Speichern der Ergebnisse
        file_name = f"adapter_summary_{self.base_model_name}_{self.start_time}_{config_name}_{self.mapping_type}.csv"
        summarize_results(self.results_summary, file_name)
        print(self.model.adapter_summary())


    def train_fusion_layer(self):
        """
        Führt das Training eines Adapter-Fusion-Modells durch.
        """
        print("\nTrainiere Fusion Layer...")
        adapter_setup = Fuse(*self.trained_adapters)
        self.model.add_tagging_head("head_fusion", num_labels=len(self.id_to_label), id2label=self.id_to_label)
        self.model.add_adapter_fusion(adapter_setup)
        self.model.set_active_adapters(adapter_setup)
        self.model.train_adapter_fusion(adapter_setup)

        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)

        run_id = wandb.util.generate_id()
        run = wandb.init(
            project="Cross-Corpus-NER",
            name=f"Fusion_{run_id}",
            tags=[f"timestamp_{self.start_time}", "Fusion"],
            reinit=True,
            resume=False,
            id=run_id,
        )

        lr = getattr(wandb.config, "learning_rate", None) or self.config.get("learning_rate", 2e-4)
        batch_size = getattr(wandb.config, "batch_size", None) or self.config.get("batch_size", 8)
        epochs = getattr(wandb.config, "num_train_epochs_fusion", None) or self.config.get("num_train_epochs_fusion", 1)

        config_name = getattr(wandb.config, "adapter_config_name", None) or self.config.get("adapter_config_name", "houlsby")
        
        # Kombinierte Trainings- und Validierungsdaten der einzelnen Korpora
        cross_train = concatenate_datasets([self.tokenized_datasets[ds]["train"] for ds in self.tokenized_datasets])
        cross_eval = concatenate_datasets([self.tokenized_datasets[ds]["dev"] for ds in self.tokenized_datasets])

        training_args = TrainingArguments(
            output_dir=os.path.join(self.base_adapter_dir, "results", "fusion"),
            evaluation_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir="./logs/fusion",
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="wandb",
            local_rank=-1,
        )

        trainer = AdapterTrainer(
            model=self.model,
            args=training_args,
            train_dataset=cross_train,
            eval_dataset=cross_eval,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=lambda pred: calculate_metrics(pred, self.id_to_label),
        )

        trainer.train()
        trainer.evaluate()

        self.model.save_adapter_fusion(os.path.join(self.base_adapter_dir, "adapters", "fusion"), adapter_setup)
        self.model.save_head(os.path.join(self.base_adapter_dir, "heads", "head_fusion"), "head_fusion")

        fusion_summary = evaluate_model_on_testsets(
            trainer=trainer,
            adapter_name="fusion",
            id_to_label=self.id_to_label,
            tokenized_datasets=self.tokenized_datasets,
            base_model_name=self.base_model_name,
            adapter_config_name=config_name,
            mapping_type=self.mapping_type,
        )

        run.finish()
        file_name = f"fusion_summary_{self.base_model_name}_{self.start_time}_{config_name}_{self.mapping_type}.csv"
        summarize_results(fusion_summary, file_name)
        print(self.model.adapter_summary())


if __name__ == "__main__":
    print("Führe AdapterTrainerManager aus...")

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
    manager = AdapterTrainerManager(
        base_model_name=base_model_name,
        mapping_type=mapping_type,
        config=global_config,
        tokenized_datasets=pre.get_tokenized_datasets(),
        tokenizer=pre.tokenizer,
        data_collator=pre.data_collator,
        id_to_label=pre.id_to_label,
    )

    # Starte Training
    manager.train_all_adapters()
    manager.train_fusion_layer()