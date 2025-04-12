import os
import wandb
import yaml
import warnings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "..", "config", "run_config.yaml")

def read_yaml_config(file_path=FILE_PATH):
    """
    Liest die Konfiguration aus der YAML-Datei und gibt diese als Dictionary zurück.
    Falls die Datei nicht existiert, wird eine Warnung ausgegeben und ein leeres Dictionary
    zurückgegeben.
    """

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"Konfiguration aus {file_path} geladen.")
        return config
    else:
        warnings.warn(f"Die Konfigurationsdatei '{file_path}' existiert nicht. Bitte generiere die Konfiguration.")
        return {}

def load_config():
    """
    Lädt die Sweep-Konfiguration und gibt wandb.config zurück. 
    Diese Funktion verwendet den Offline-Modus, um zu verhindern, 
    dass der Sweep-Ausführungskontext betreten wird und löscht dann die Sweep-Umgebungsvariablen, 
    damit nachfolgende wandb.init-Aufrufe neu beginnen.
    """
    # Starte wandb in offline mode damit nicht in sweep context gelogged wird.
    unique_run_id = wandb.util.generate_id()
    run = wandb.init(
        project="Cross-Corpus-NER", 
        name="config_load",
        reinit=True,
        mode="offline",
        resume=False,
        id=unique_run_id,
    )
    config = wandb.config
    save_config_to_yaml(config)
    run.finish()
    
    for key in ["WANDB_SWEEP_ID", "WANDB_RUN_ID", "WANDB_RESUME"]:
        os.environ.pop(key, None)

def save_config_to_yaml(config, filename=FILE_PATH):
    """
    Speichert die übergebene Konfiguration in eine YAML-Datei.
    Falls die Konfiguration leer ist, wird eine Warnmeldung ausgegeben,
    und das bestehende File wird nicht überschrieben.
    """
    # Wandle wandb.config in ein dict um
    config_dict = dict(config)
    
    # Überprüfe, ob die Konfiguration leer ist
    if not config_dict:
        warnings.warn(f"Keine Sweep-Konfiguration gefunden. Überschreibe {filename} nicht.")
        return

    # Schreibe die Konfiguration in die YAML-Datei
    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(config_dict, file, default_flow_style=False, allow_unicode=True)
    print(f"Konfiguration wurde in {filename} gespeichert.")

if __name__ == "__main__":
    load_config()