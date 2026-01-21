# DeepTox — Toxicity Classifier

Short project README generated from the workspace.

**Project**: DeepTox (Toxicity classification using Keras)

**What it is**
- A small text (and audio) toxicity classifier implemented with Keras. The repository contains training code (`train.py`), a model file (`toxicity.h5`) tracked with Git LFS, and a Gradio-based demo UI in `app.py` that accepts text or audio and shows predicted toxicity probabilities.

**Key files**
- `app.py`: Gradio app to run the model (loads `toxicity.h5` and `tokenizer.pkl`).
- `train.py`: training script that reads `train.csv`, prepares a tokenizer, and trains a Keras bidirectional LSTM model; writes `toxicity.h5` and `tokenizer.pkl`.
- `requirements.txt`: Python dependencies.
- `toxicity.h5`: trained model (stored with Git LFS). See `.gitattributes` for LFS tracking rules.
- `tokenizer.pkl`: tokenizer used to convert text to sequences.

Getting started
1. Clone the repository:
   ```powershell
   git clone https://github.com/SatyamGupta32/DeepTox.git
   cd DeepTox
   ```
2. Create and activate a virtual environment (PowerShell):
   ```powershell
   python -m venv .venv
   & .\\.venv\\Scripts\\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   git lfs install
   git lfs pull
   ```

Running the demo
- Run the Gradio demo UI (loads `toxicity.h5` and `tokenizer.pkl`):
  ```powershell
  & .\\.venv\\Scripts\\Activate.ps1
  python app.py
  ```
  The app launches a local web UI and (if `share=True` in `app.py`) may expose a temporary public link.

Training
- `train.py` expects a CSV dataset with one text column (contains `comment` or `text` in the name) and one or more binary label columns (e.g. `toxic`, `insult`, `obscene`). See the top of `train.py` for configurable constants.
- To train locally (small quick run):
  ```powershell
  & .\\.venv\\Scripts\\Activate.ps1
  python train.py
  ```
  This will produce `tokenizer.pkl` and `toxicity.h5`.

Notes about large files / Git LFS
- The repository uses Git LFS for large binaries. `.gitattributes` contains rules such as `*.h5 filter=lfs` and `train.csv` is also tracked by LFS.
- When cloning, run `git lfs install` then `git lfs pull` to fetch large objects.
- GitHub LFS storage may have bandwidth/storage quotas on free accounts.

Security & privacy
- `toxicity.h5` and `train.csv` may contain model weights and training data — be mindful of privacy and licensing of any data you include.

If you want me to:
- add example inputs, screenshots, or badges — say which details to include, or
- produce a short `CONTRIBUTING.md` or `LICENSE` file.

---
Generated from repository files: `app.py`, `train.py`, `requirements.txt`, `.gitattributes`.