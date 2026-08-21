# XOR — DigitalOcean: Constructing Neural Networks From Scratch (Part 1)

**Tutorial:** [Constructing Neural Networks From Scratch: Part 1](https://www.digitalocean.com/community/tutorials/constructing-neural-networks-from-scratch)

NumPy-only XOR network: 2 inputs → 2 hidden (sigmoid) → 1 output (sigmoid), binary cross-entropy, gradient descent for 10 000 epochs.

```bash
# from repo root
source .venv/bin/activate   # or: python3 -m venv .venv && source .venv/bin/activate
pip install -r XOR/requirements.txt
python XOR/xor_from_scratch.py
```

**Status:** trained locally — XOR cases match (0, 1, 1, 0). Loss plot: `xor_loss.png`.

The Keras AND/XOR section at the end of the tutorial is skipped here on purpose.
