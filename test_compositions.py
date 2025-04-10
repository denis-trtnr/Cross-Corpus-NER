import os
import evaluate
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix
from transformers import TrainingArguments
from adapters import AdapterTrainer, AutoAdapterModel
import adapters.composition as ac

# Importiere den DataPreprocessor
from data_preprocessing import DataPreprocessor
# Importiere metrics utils aus dem Metrics-Modul.
from metrics_utils import summarize_results
from config_utils import read_yaml_config

class AdapterCompositionEvaluator:
    """
    Klasse zur Evaluation verschiedener Adapter-Kompositionen (z.B. Stack, Average)
    auf einem gegebenen Modell und tokenisierten Datensätzen.
    """

    def __init__(
        self,
        base_model_name,
        mapping_type,
        tokenizer,
        tokenized_datasets,
        data_collator,
        id_to_label,
        adapter_names,
        base_adapter_dir="/netscratch/dtrautner/studienarbeit/results",
    ):
        """
        Initialisiert die Evaluationsumgebung.

        Parameter:
        - base_model_name (str): Modellname
        - mapping_type (str): Mapping-Strategie
        - model: Modellobjekt mit geladenen Adaptern
        - tokenizer: Genutzter Tokenizer
        - tokenized_datasets (dict): Dict mit test-Datasets
        - data_collator: Collator für DataLoader
        - id_to_label (dict): Mapping von Label-IDs zu Labelnamen
        - adapter_names (list): Liste der zu kombinierenden Adapternamen
        - base_adapter_dir (str): Basispfad zu gespeicherten Adaptern und Heads
        - metric (evaluate.Metric): Eval-Metrik (default: seqeval)
        """
        self.base_model_name = base_model_name
        self.mapping_type = mapping_type
        self.model = AutoAdapterModel.from_pretrained(base_model_name)
        self.tokenizer = tokenizer
        self.tokenized_datasets = tokenized_datasets
        self.data_collator = data_collator
        self.id_to_label = id_to_label
        self.adapter_names = adapter_names
        self.base_adapter_dir = base_adapter_dir
        self.metric = evaluate.load("seqeval")
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    def load_stored_adapters(self):
        """
        Lädt alle angegebenen Adapter und ihre Heads aus dem Dateisystem ins Modell.
        """
        for name in self.adapter_names:
            adapter_path = os.path.join(self.base_adapter_dir, "adapters", name)
            head_path = os.path.join(self.base_adapter_dir, "heads", name)
            print(f"Lade Adapter {name} ...")
            self.model.load_adapter(adapter_path, load_as=name)
            print(f"Lade Head {name} ...")
            self.model.load_head(head_path, load_as=name)


    def evaluate_composition(self, method_name, composition_obj):
        """
        Setzt die aktive Komposition im Modell, evaluiert auf allen Testsets und
        gibt eine Ergebnisübersicht zurück.
        """

        # Setze die aktive Adapter-Komposition im Modell
        self.model.set_active_adapters(composition_obj)
        
        results = []

        # Iteriere über alle Testsets in tokenized_datasets
        for test_dataset_name, dataset in self.tokenized_datasets.items():
            print(f"Evaluierung mit {method_name} auf {test_dataset_name}-Testset...")
            
            # Initialisiere einen AdapterTrainer ausschließlich für die Vorhersage
            training_args = TrainingArguments(
                output_dir="./results",
                per_device_eval_batch_size=8,
                evaluation_strategy="no"  # Es findet kein Training statt
            )

            trainer = AdapterTrainer(
                model=self.model,
                args=training_args,
                eval_dataset=dataset["test"],
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
            )

            # Vorhersagen generieren
            predictions_output = trainer.predict(dataset["test"])

            # Extrahiere Logits und Label-IDs aus dem predictions_output-Objekt
            logits = predictions_output.predictions
            labels = predictions_output.label_ids

            # Jetzt kann np.argmax angewendet werden
            predictions = np.argmax(logits, axis=-1)
            true_labels = [[self.id_to_label[l] for l in label if l != -100] for label in labels]
            pred_labels = [
                [self.id_to_label[p] for (p, l) in zip(pred, label) if l != -100]
                for pred, label in zip(predictions, labels)
            ]
            # Berechne seqeval-Metriken
            overall_metrics = self.metric.compute(predictions=pred_labels, references=true_labels, zero_division=1)

            # Flache die Labels und Vorhersagen für die Confusion-Matrix
            true_labels_flat = [label for sublist in true_labels for label in sublist]
            pred_labels_flat = [pred for sublist in pred_labels for pred in sublist]

            # Reduziere die Klassen auf tatsächlich vorkommende Labels
            true_flat = set(true_labels_flat)
            pred_flat = set(pred_labels_flat)
            relevant_classes = sorted(set(true_flat).union(set(pred_flat)))

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
        compositions_summary_file_name= f"{method_name}_summary_{self.base_model_name}_{self.start_time}.csv"
        summarize_results(results, compositions_summary_file_name)
        return results


    def test_all_compositions(self):
        """
        Testet alle Kompositionsmethoden (außer Fuse) und gibt eine Übersicht der Ergebnisse aus.
        """
        # Lade alle Adapter und ihre Heads
        self.load_stored_adapters()

        # Definiere die Kompositionsmethoden, die getestet werden sollen.
        # Hier: "stack", "average" und "parallel".
        compositions = {
            "stack": ac.Stack(*self.adapter_names),
            "average": ac.Average(*self.adapter_names, weights=[0.2, 0.2, 0.2, 0.2, 0.2])
        }
        
        all_results = []
        
        for method, comp_obj in compositions.items():
            print(f"\nTeste Komposition: {method}")
            results = self.evaluate_composition(method, comp_obj)
            all_results.extend(results)


        compositions_summary_file_name= f"compositions_summary_{self.base_model_name}_{self.start_time}.csv"
        summarize_results(all_results, compositions_summary_file_name)

        # Deaktiviere die aktiven Adapter nach der Evaluation
        self.model.set_active_adapters(None)

if __name__ == "__main__":
    print("Führe AdapterTrainerManager aus...")

    # Konfiguration laden
    global_config = read_yaml_config()

    # Werte aus der Konfiguration extrahieren
    base_model_name = global_config.get(
        "model", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )
    mapping_type = global_config.get("mapping_type", "granular")

    adapter_names = [
        "SETH_adapter",
        "Variome_adapter",
        "Variome120_adapter",
        "Amia_adapter",
        "TmVar_adapter"
    ]

    # Erstelle Preprocessor
    pre = DataPreprocessor(
        mapping_type=mapping_type,
        base_model=base_model_name,
    )

    # Erstelle eine Instanz des AdapterCompositionEvaluator
    evaluator = AdapterCompositionEvaluator(
        base_model_name=base_model_name,
        mapping_type=mapping_type,
        tokenizer=pre.tokenizer,
        tokenized_datasets=pre.get_tokenized_datasets(),
        data_collator=pre.data_collator,
        id_to_label=pre.id_to_label,
        adapter_names=adapter_names,
    )

    # Starte Training
    evaluator.test_all_compositions()
