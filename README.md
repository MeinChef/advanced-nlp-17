# advanced-nlp-17
Repository for the Advanced NLP Block-Course at Osnabrück University.

---

# Getting Started
Clone the Repository

```bash
git clone https://github.com/MeinChef/advanced-nlp-17.git
```

Set up and activate a python environment

```bash
conda create -n advnlp python=3.12
conda activate advnlp
```
**--- OR ---** 
```bash
python -m venv /path/to/new/virtual/environment
```
Navigate to the repository folder
```bash
cd advanced-nlp-17
```

And execute setup script
```bash
python setup.py
```

You should be all set to execute the main programs for each task!

---

# Task 1
Task one was concerned with investigating scaling laws (Hoffman et al, 2022, https://doi.org/10.48550/arXiv.2203.15556).
For this we trained four models on each of the four partial datasets.
|    | Layers | Heads | Embedding  | Params     | Dataset-Parts |
|----|--------|-------|------------|-----------:|---------------|
| XS | 2      | 2     | 64*2 = 128 |    119.168 | 0.125         |
| S  | 4      | 4     | 64*4 = 256 |  3.230.208 | 0.25          |
| M  | 5      | 5     | 64*5 = 320 |  6.250.240 | 0.5           |
| L  | 6      | 6     | 64*6 = 384 | 10.745.088 | 1             |

We trained every model for a fixed amount of iterations.
Since `batch_size` and `block_size` are fixed across trials, we know that with every batch (64 * 256) the model has seen 16384 tokens.
The amount of batches (iterations) can then be calculated with `params / 16384 * 20` ([code](chinchilla.py?plain=1#L116)).

The terminal output will get written into `train.out` and `sample.out` within folders with of the naming scheme `logs/out-shakespeare-{layers}-{embed}-{dataset_split}`.
From there we created visualisations (see [part_X_vis_TASK.ipynb](part_1_vis_Scaling_Laws.ipynb)).

To see all of this in action, execute
```bash
python chinchilla.py
```
and the aforementioned Jupyter Notebook.

# Task 2
TODO

# Task 3
TODO
