# micrograd (Karpathy)

**Source:** [karpathy/micrograd](https://github.com/karpathy/micrograd)  
Companion talk: [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)

This is not NumPy matrix backprop. Each number is a `Value` that records `+`, `*`, `**`, and `relu` on a graph. `backward()` walks that graph (reverse-mode autodiff). `Neuron` / `Layer` / `MLP` sit on top, PyTorch-style, still scalar-by-scalar.

The Python package is the inner `micrograd/` folder (`engine.py` ~ autograd, `nn.py` ~ net). Not `pip install micrograd` — this folder *is* the engine.

```bash
# from repo root
source .venv/bin/activate
pip install -r micrograd/requirements.txt

# 1) README autograd check (should print 24.7041, 138.8338, 645.5773)
python micrograd/demo_autograd.py

# 2) 2→16→16→1 MLP on sklearn moons (slow: scalar ops, ~a minute)
python micrograd/demo_moons.py
```

**Status:** autograd README check passed. Moons MLP reached 100% train acc (100 points); `moons.png` written.
