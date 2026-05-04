# advanced-nlp-17
Repository for the Advanced NLP Block-Course at Osnabrück University.

We used the nanoGPT repository at https://github.com/karpathy/nanogpt as the model and build upon it.

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

If you have a **blackwell-series GPU**, which is not yet supported by the stable torch, you might need to install the [nightly torch version](https://pytorch.org/get-started/locally/). 
The setup script informs you in the last couple of lines if that is applicable for you.
We used the following command:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

You should now be all set to execute the main programs for each task!

---

# Task 1 (Pre-Training)
Task one was concerned with investigating scaling laws (Hoffman et al., 2022, https://doi.org/10.48550/arXiv.2203.15556).
For this we trained four models on fractions of the shakespeare-char dataset provided by nanoGPT.
Each model was trained on all dataset fractions.
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
and the visualisations in the aforementioned Jupyter Notebook.

# Task 2 (Supervised Fine-Tuning)
For Task two we pre-processed the full dataset for speaker prediction and prose/verse classification.
We then fine-tuned our best model (M trained on the full dataset) on these datasets for each individual task and for both.
For that we needed to add our special tokens to the embedding layer.

To run this task, run the following:
```bash
python part_2_main.py
```

### Speaker Prediction
Each speaker (in UPPERCASE) gets replaced with the token `@`, and after the lines the speaker spoke, we place the solution in `<>`.

An example would look like this:
> @:    
> Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ... \<ROMEO\>

### Prose/Verse Classification
Our heuristic for classifying whether the block of text is prose or a verse, we used vowels (or syllables).
If each line in the block was roughly $10\pm3$ syllables, it's a verse, else it's prose.
As an indicator that this is a classification task, we used the `|` symbol, and `<>` to embed the answer.

An example would look like this:
> ROMEO:
> Lorem ipsum dolor sit amet
> consectetur adipiscing elit
> sed do eiusmod tempor incididunt | <VERSE>


# Task 3
TODO
