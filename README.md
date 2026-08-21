# nns

Neural nets from scratch in **NumPy** (no PyTorch / TensorFlow). Tutorials live under `practice/`; small game agents live at the repo root.

## Setup

```bash
cd nns
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy matplotlib scikit-learn
```

Run each project from **its own directory** so local imports resolve:

```bash
cd sudoku   # or snake / tic-tac-toe / practice/...
python train.py
```

Generated `dataset.npz` / `weights*.npz` files are gitignored; train locally to create them.

## Layout

| Path | What |
|------|------|
| `practice/XOR` | Tiny MLP learns XOR |
| `practice/letters` | Letter classification from scratch |
| `practice/cardiovascular-risk` | Tabular risk model from scratch |
| `practice/digits` | Digit classification from scratch |
| `practice/micrograd` | Tiny autograd + moon demo (Karpathy-style) |
| `tic-tac-toe/` | Board → minimax labels → MLP policy |
| `snake/` | Heuristic-cloned then RL-improved Snake net |
| `sudoku/` | Cell-digit MLP + curriculum + iterative solve |

---

## Practice

```bash
cd practice/XOR && python xor_from_scratch.py
cd practice/letters && python letters_from_scratch.py
cd practice/cardiovascular-risk && python cardio_from_scratch.py
cd practice/digits && python digits_from_scratch.py
cd practice/micrograd && python demo_autograd.py
cd practice/micrograd && python demo_moons.py
```

---

## Tic-tac-toe

MLP maps a board (`1 / -1 / 0`) to a move; trained on minimax labels with illegal-move masking.

```bash
cd tic-tac-toe
python dataset.py    # build labels
python train.py      # → weights.npz
python eval.py       # vs minimax
python play.py       # play in the terminal
```

---

## Snake

11-D state → U/D/L/R. Dataset from a heuristic teacher; optional REINFORCE in `improve.py`.

```bash
cd snake
python dataset.py
python train.py
python eval.py
python play.py                 # watch the net
python play.py --heuristic     # watch the teacher
python improve.py              # optional RL fine-tune
```

---

## Sudoku

Partial grid + focus cell → digit 1–9. Features (**927-D**): board one-hots, focus, row/col/box presence, legal candidates. Trained with mid-solve examples, legal-only CE, confidence-thresholded fill + light backtrack.

**Best weights:** `weights_curriculum.npz` (train via curriculum, not plain `train.py`).

```bash
cd sudoku

# train curriculum 20 → 35 → 45 → 55 holes (~15s)
python curriculum.py

# scoreboard (reuse saved weights, no retrain)
python -c "
from train import load_weights
from curriculum import WEIGHTS_PATH, eval_curriculum
eval_curriculum(load_weights(WEIGHTS_PATH))
"

# harder unique puzzles
python solve_loop.py --weights weights_curriculum.npz --holes 50

# visual board + mistakes
python demo.py --weights weights_curriculum.npz --holes 35
```

### Results (curriculum net)

| Empties | Unique exact solve |
|--------:|:------------------:|
| 20 | 4/4 |
| 35 | 4/4 |
| 45 | 4/4 |
| 50 | ~3/5 (`solve_loop`) |
| 55 | 0/4 |

Easy/medium unique puzzles are solid; ~55 empties still needs real search beyond the current greedy+τ loop.

---

## Notes

- Everything is intentional toy scale: readable code over SOTA accuracy.
- Local `**/plan.md` files are gitignored planning scratchpads.
