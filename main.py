from data_preprocessing import DataPreprocessor
from test_compositions import AdapterCompositionEvaluator
from train_adapters import AdapterTrainerManager
from train_baseline import BaselineTrainerManager
from config_utils import read_yaml_config

# Konfiguration laden
global_config = read_yaml_config()

adapter_names = global_config.get("adapter_names")

def evaluate(mapping_type, base_model_name):
    print(f"\n🔧 Starte Training und Evaluation für Modell: {base_model_name} | Mapping: {mapping_type}\n")

    # Erstelle Preprocessor
    pre = DataPreprocessor(
        mapping_type=mapping_type,
        base_model=base_model_name,
    )

    # Tokenisiere die Datensätze
    tokenized_datasets=pre.get_tokenized_datasets()


    # Erstelle eine Instanz des AdapterTrainerManagers
    adapter_manager = AdapterTrainerManager(
        base_model_name=base_model_name,
        mapping_type=mapping_type,
        model_config=pre.config,
        config=global_config,
        tokenized_datasets=tokenized_datasets,
        tokenizer=pre.tokenizer,
        data_collator=pre.data_collator,
        id_to_label=pre.id_to_label,
    )

    # Starte Training
    adapter_manager.train_all_adapters()
    adapter_manager.train_fusion_layer()


    # Erstelle eine Instanz des AdapterCompositionEvaluator
    evaluator = AdapterCompositionEvaluator(
        base_model_name=base_model_name,
        mapping_type=mapping_type,
        tokenizer=pre.tokenizer,
        tokenized_datasets=tokenized_datasets,
        data_collator=pre.data_collator,
        id_to_label=pre.id_to_label,
        model_config=pre.config,
        adapter_names=adapter_names,
    )

    # Starte Evaluation
    evaluator.test_all_compositions()

    # Erstelle eine Instanz des BaselineTrainerManagers
    baseline_manager = BaselineTrainerManager(
        base_model_name=base_model_name,
        mapping_type=mapping_type,
        model_config=pre.config,
        config=global_config,
        tokenized_datasets=tokenized_datasets,
        tokenizer=pre.tokenizer,
        data_collator=pre.data_collator,
        id_to_label=pre.id_to_label,
    )

    # Starte Baseline Training
    baseline_manager.train_all_models()

    
    

if __name__ == "__main__":
    base_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    mapping_type = global_config.get("mapping_type", "granular")

    evaluate(mapping_type, base_model_name)