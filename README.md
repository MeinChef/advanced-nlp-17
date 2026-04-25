# advanced-nlp-17
Repository for the Advanced NLP Block-Course at Osnabrück University.

---

# Getting Started
Clone the Repository

```bash
git clone https://github.com/MeinChef/advanced-nlp-17.git
```

Set up and activate a conda environment

```bash
conda create -n advnlp python=3.12
conda activate advnlp
```

```bash
pytorch for cuda if you want to use a Blackwell gen card
-> use requirements_blackwell.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

Execute setup script
```bash
python setup_venv.py
or
python setup_conda.py
```