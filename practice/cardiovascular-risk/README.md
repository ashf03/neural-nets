# cardiovascular-risk — Neural network from scratch (Stackademic)

**Tutorial:** [Learn to Build a Neural Network From Scratch — Yes, Really.](https://blog.stackademic.com/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc)

Toy **cardiovascular risk** classifier: weight (lb) + height (in) → 0/1. Architecture **2 → 3 → 3 → 1**, sigmoid, binary cross-entropy, vectorized backprop (the article’s section 11).

Inputs are **standard-scaled** (article challenge 2). Raw 150/70 values saturate sigmoid and barely train.

```bash
# from repo root
source .venv/bin/activate
pip install -r cardiovascular-risk/requirements.txt
python cardiovascular-risk/cardio_from_scratch.py
```

Watch cost drop every 20 epochs, then the ten true vs predicted labels. Plot: `cost.png`.

**Status:** trained locally — all 10 labels match. Final cost ~0.23.
