# letters — Neural network from scratch (GeeksforGeeks)

**Tutorial:** [Implementation of neural network from scratch using NumPy](https://www.geeksforgeeks.org/numpy/implementation-of-neural-network-from-scratch-using-numpy/)

Classifies 5×6 binary “images” of **A, B, and C**. Architecture: 30 inputs → 5 hidden (sigmoid) → 3 outputs (sigmoid), trained with MSE and backprop.

```bash
# from repo root
source .venv/bin/activate
pip install -r letters/requirements.txt
python letters/letters_from_scratch.py
```

That run prints per-epoch accuracy, then predicts A, B, and C. Plots land in this folder: `letters.png`, `accuracy.png`, `loss.png`.

**Status:** trained locally — A, B, and C all classified correctly.
