"""README example from https://github.com/karpathy/micrograd"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f"{g.data:.4f}")  # 24.7041
g.backward()
print(f"{a.grad:.4f}")  # 138.8338
print(f"{b.grad:.4f}")  # 645.5773
