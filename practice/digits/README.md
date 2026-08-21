# digits — Neural network from scratch (Let's Data Science)

**Tutorial:** [Build a Neural Network from Scratch in Python](https://letsdatascience.com/blog/build-a-neural-network-from-scratch-in-python)

Classifies sklearn **digits** (1,797 images, 8×8 pixels → 10 classes). Architecture **64 → 128 → 64 → 10**: He init, ReLU, softmax, cross-entropy, mini-batch SGD. The tutorial reports ~97.8% val accuracy.

Needs scikit-learn for the dataset and `StandardScaler` (the article says unscaled pixels stall around ~90%).

```bash
# from repo root
source .venv/bin/activate
pip install -r digits/requirements.txt
python digits/digits_from_scratch.py
```

200 epochs — this one takes longer than XOR / letters / cardio. You should see a print every 50 epochs, then a final val accuracy and `training.png`.

**Status:** trained locally — 97.50% validation accuracy (351/360). Tutorial quoted 97.78%; a few-sample split difference is normal.
