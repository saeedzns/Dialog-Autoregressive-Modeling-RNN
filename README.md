# Character-Level RNN for Dialog Autoregressive Modeling (Pure JAX)

A clean, self-contained JAX implementation of a **character-level vanilla RNN language model** trained on the **DailyDialog** dataset.  
The project demonstrates autoregressive text generation using fundamental recurrent network mechanics — **no Flax, no Optax, no shortcuts** — just raw JAX and a stubborn insistence on understanding what’s going on under the hood.

This notebook was developed and tested entirely in **Google Colab**, so you can open it and run everything without any local setup.

---

## What the Model Does

**Input:**  
- Multi-turn human conversations from the **DailyDialog** dataset.  
- Text is normalized, lowercased, and converted into sequences of integer-encoded characters.  
- Each sequence represents one or more dialogue turns.

**Process:**  
- The **RNN** reads each character sequentially, maintaining a hidden state that represents context.  
- At every step, it predicts the probability distribution of the next character using a softmax output layer.  
- Training minimizes cross-entropy loss between predicted and actual characters.  
- The model learns to mimic conversational patterns, tone, and punctuation through repetition.

**Output:**  
- Quantitative metrics: **loss** and **perplexity** for training, validation, and test sets.  
- Qualitative results: automatically generated dialogues where each character is predicted by the model itself, such as:  
  ```
  ① hi, how are you today?
  ② i'm doing well, thanks for asking!
  ① good to hear that.
  ```
- Visualization: training curves for loss and learning rate, showing learning stability over epochs.

---
## Environment

Everything runs on Colab’s standard runtime.  
If you prefer local execution, install dependencies manually:

```bash
pip install --upgrade "jax[cpu]" jaxlib matplotlib kagglehub
```

For GPU acceleration:

```bash
pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

> The notebook automatically sets basic XLA threading flags to prevent Colab CPUs from overheating like cheap toasters.

---

## Dataset

The model uses the [**DailyDialog**](https://www.kaggle.com/datasets/thedevastator/dailydialog-multi-turn-dialog-with-intention-and) dataset (multi-turn human conversations), fetched automatically through **KaggleHub**:

```python
import kagglehub
DATA_DIR = kagglehub.dataset_download(
    "thedevastator/dailydialog-multi-turn-dialog-with-intention-and"
)
```

After download, the notebook constructs:
- `train.csv`
- `validation.csv`
- `test.csv`

### Preprocessing Steps

- Normalize punctuation and casing.  
- Inject speaker symbols (`①`, `②`) at character level.  
- Add control symbols:
  - `␂` = BOS  
  - `␞` = EOS  
  - `¤` = UNK  

A compact **character vocabulary** is learned from the training split, and dialog text is chunked into fixed-length sequences of integer tokens.

---

## Model Overview

A simple **vanilla RNN** implemented directly in JAX:

- Hidden update:  
  h_t = tanh(LayerNorm(W_xh x_t + W_hh h_{t-1} + b_h))
- Optional **LayerNorm**, **tied embeddings**, and **variational dropout**  
- **Manual Adam** optimizer and **cosine learning-rate schedule**  
- **EMA (Exponential Moving Average)** parameters for stable evaluation  
- **Top-p sampling** and **beam-search decoding** for generation

Key sections of the notebook:
- Parameter initialization  
- RNN step and forward pass  
- Training loop with loss regularization and gradient clipping  
- Sampling utilities for text generation  
- Visualization of training curves (loss, perplexity, learning rate)

---

## How to Use

1. **Open the notebook in Colab:**  
   [Open in Colab](https://colab.research.google.com/drive/1ewN-Z-eLGEd52Yy35Ua6idGHwPHi6D3t?usp=sharing)

2. **Run all cells.**  
   The notebook will:
   - Download the dataset automatically  
   - Preprocess and tokenize the text  
   - Train the RNN with cosine LR and EMA  
   - Report train/validation/test losses and perplexities  
   - Generate new dialog text via top-p sampling or beam search

3. **Try text generation:**

```python
# Top-p sampling
s = generate_text(params, "Hi, ", max_len=120)
print(s)

# Beam search decoding
b = beam_search_decode(params, "Why are", beam_size=4)
print(b)
```

---

## Results

The notebook prints:
- Final training, validation, and test losses  
- Perplexity for raw and EMA parameters  
- Training curves for loss and learning rate  

EMA weights generally yield more stable and coherent outputs.

---

## Project Layout

```
.
├── NNDS_2024_Final_Homework_Oct.ipynb   # Main Colab notebook
└── README.md
```

## Notes

- The implementation intentionally avoids high-level libraries to expose the math.  
- CPU runs are fine; GPU is faster but not required.  
- Character-level modeling trades a smaller vocabulary for slower learning, but it never breaks on unseen words.  
- This project doubles as a transparent study case for recurrent neural architectures written purely in JAX.

---

## License

Educational use only. If you reuse this code, please provide attribution.

---

## Acknowledgments

- **DailyDialog** dataset authors  
- **JAX** development team for the minimalist numerical stack that made this experiment possible  
