# snake

From-scratch NumPy net (state → U/D/L/R). See local `plan.md` (gitignored).

```bash
# from repo root
source .venv/bin/activate
python snake/game.py
python snake/encode.py
python snake/heuristic.py
python snake/dataset.py
python snake/nn.py
python snake/train.py
python snake/eval.py
python snake/play.py                  # watch the net
python snake/play.py --policy heuristic
python snake/play.py --policy random --delay 0.05
python snake/improve.py               # REINFORCE fine-tune → weights_improved.npz
python snake/play.py --weights snake/weights_improved.npz
```

Weights → `weights.npz` / `weights_improved.npz` (gitignored).
