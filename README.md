# Cross-Corpus Named Entity Recognition

Welcome to the **Cross-Corpus-NER** project! This repository contains the code and configurations used for a study thesis exploring the generalization capabilities of NER models in the biomedical domain by using Adapter-based architectures for transfer learning

> 📚 **This work was conducted as part of a study in collaboration with the [DFKI Speech & Technology Lab](https://www.dfki.de/en/web/research/research-departments/speech-and-language-technology)**



## 🗂️ Overview

```bash
Cross-Corpus-NER/
├── config/
│   ├── environment.yaml        # 📦 Conda enviroment used during training
│   ├── run_configy.yaml        # ⚙️ Configs that are used before wandb init 
│   └── sweep_config.yaml       # 🎯 Weights & Biases hyperparameter sweep configuration
├── data/                       # 📁 Directory for storing datasets and related files
├── docs/                       # 📄 Submitted study thesis and research proposal documents
├── src/
│   ├── main.py                 # 🚀 Entry point script to launch training with selected configs and model types
│   ├── data_preprocessing.py   # 🧹 Handles dataset preprocessing: formatting, label alignment, tokenization
│   ├── train_baseline.py       # 🧪 Training script for training of standard transformer-based models (no adapters)
│   ├── train_adapters.py       # 🧠 Training script for models using AdapterHub modules
│   ├── test_compositions.py    # 🧬 Run experiments on different adapter composition strategies
│   ├── config_utils.py         # 🛠️  Helper functions for loading configs and parsing args
│   ├── mapping_utils.py        # 🔄 Function for label/tagset harmonization across datasets
│   ├── metrics_utils.py        # 📊 Evaluation functions & creation of visualizations
│   └── baseline.ipynb          # 📓 Notebook for quick tests or prototyping of the baseline model setup
├── requirements.txt            # 📋 Lists project dependencies for installation
└── README.md                   # 📘 You're looking at it :-)

```

## 🔬 Datasets Used

The following mutation datasets were used for training and cross-corpus evaluation:

-  [**SETH**](https://github.com/rockt/SETH) by Verspoor et al. (2016)
-  [**Variome**](https://bitbucket.org/readbiomed/variome-corpus-data) by Thomas et al. (2013)
-  [**Variome120**](https://github.com/Rostlab/nala/tree/develop/resources/corpora/variome_120) by Jimeno & Verspoor (2014)
-  [**AMIA**](https://github.com/ibm-aur-nlp/amia-18-mutation-corpus) by Jimeno et al. (2018)
-  [**TmVar**](https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/tmVar.html) by Wei et al. (2013)


## ⚠️ Environment-Specific Notes

This repository contains code components that are specific to the infrastructure used during development:

- ✅ [W&B](https://wandb.ai) for experiment tracking and automated sweeps
- 🛰️ [Pegasus](https://pegasus.dfki.de/) Cluster-specific scripts (e.g., SLURM configs, paths)

If you're running this outside the original setup, you may need to:

- Add W&B API Key to your setup 
- Customize or disable `sweep.yaml` if not using W&B  
- Manually modify storage/output locations and env files



## 🧪 Example Usage

#### On Cluster Using [Pegasus Bridle](https://github.com/DFKI-NLP/pegasus-bridle)
```bash
wandb sweep config/sweep_config.yaml
```
```bash
bash /home/dtrautner/dev/pegasus-bridle/wrapper.sh wandb agent denistrautner-dhbw-duale-hochschule-baden-w-rttemberg/Cross-Corpus-NER-src/9wn48tik
```

or without w&b:

```bash
wandb bash /home/dtrautner/dev/pegasus-bridle/wrapper.sh python src/main.py
```


If you want to use notebook with cluster ressources:

1. Start interactive bash session
    ```bash
    srun -K \
      --job-name jn \
      -p RTXA6000,A100-40GB,A100-80GB,H100,H100-SLT,RTX3090,batch,H200,A100-SDS,A100-PCI,V100-32GB,V100-16GB,L40S \
      --gpus=1 \
      --mem=48G \
      --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.07-py3.sqsh \
      --container-workdir=$(pwd) \
      --time 02:00:00 \
      --immediate=1000 \
      --pty bash
    ```
2. Start Jupyter Notebook Server
    ```bash
    echo "Jupyter starting at ... http://${HOSTNAME}.kl.dfki.de:8880" && jupyter notebook --ip=0.0.0.0 --port=8880 \
        --allow-root --no-browser --config /home/dtrautner/.jupyter/jupyter_notebook_config.json \
        --notebook-dir /netscratch/dtrautner/studienarbeit/notebooks
    ```
3. Copy the url from the terminal in the following format
    ```bash
    http://${HOSTNAME}.kl.dfki.de:8880/?token=<token>
    ```
4. Add Server as Kernal in VS Code


