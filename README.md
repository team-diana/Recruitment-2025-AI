# WIMBLEDIANA

![](./assets/pic.jpg)

Alberto was having a great day playing beach tennis againt his less-than-stellar CS Dept. colleagues, when suddenly a stronger opponent showed up: DIANNIK SINNER.
Alberto, which is not used to lose, but is not used to open-air activities either, knows he has no chance of winning fairly. But he has no plan of playing fair.

Help Alberto train an RL agent able to beat Diannik Sinner and win the tournament!

## Instructions

This challenge is a light‑weight **RL tennis (pong)** duel. You will **train an agent** to play the provided environment and then submit a small inference package. 
### 1) Setup (local or Colab)
```bash
pip install -r requirements.txt
```

### 2) Train a baseline (optional warm‑up)
We include a tiny DQN trainer to get you started quickly:
```bash
python src/train_dqn.py --episodes 6000 --save_path runs/checkpoint.pt
```
Consider this as a starting point to build your solution upon. You are **not** limited to DQN — any small RL (or imitation/self‑play) approach is welcome as long as you produce a **trained checkpoint** and an `agent.py` that uses it at inference time. **Remember to give to your model a cool name**.

### 3) Implement your agent interface
Create an `agent.py` that defines a class `Agent` with:
```python
class Agent:
    def __init__(self, device='cpu'): ...
    def load(self, checkpoint_path: str): ...        # load your weights
    def act(self, obs) -> int: ...                   # return 0|1|2
```
We will import this class during evaluation.

### 4) Local test
Play against a baseline bot to sanity‑check your model:
```bash
python src/evaluate_match.py --agent_a runs/checkpoint.pt --agent_b baseline_tracking
```
Alternatively, you can use `src/render_match.py` to optionally **export MP4/GIF videos** of a match for debugging or just because it is sad to not render anything.

### 5) Prepare your submission folder
Package the files described in **Expected deliverables** below. You can submit improved versions as many times as you like; only your **best** valid submission will count for the final score.

## Expected deliverables

Submit a **single folder** (or zip with the same structure) named `<Surname_Name_ModelName>/` containing:

```
<Surname_Name_ModelName>/
├─ agent.py             # implements class Agent(device='cpu'), load(), act(obs)->{0,1,2}
├─ checkpoint.pt        # your trained weights
├─ train/               # scripts or notebook used to train
├─ requirements.txt     # (optional) extra minimal deps beyond numpy/torch/imageio
├─ runs/demo.mp4        # (optional) short highlight match rendered with render_match.py
└─ other/               # (optional) any other helper files or whatever you think is useful
```

**Multiple submissions**
- You may submit **unlimited** models. We will evaluate all but only consider your **best‑performing**  one for the final score. For each model submitted, provide a single folder as described above, with a different name.

**Eligibility & penalties**

- Heuristic‑only agents (no training) are allowed for fun but will be **capped** in score compared to trained models.

## Tournament rules and Challenge Score

Max score is **1000 points**.

Each model submitted will play against all other models submitted by candidates, plus some extra ones provided by us (among which diannik_sinner) in a round-robin tournament. The final score will be determined largely by the final position in the tournament, with the max score achievable by winning the tournament. 

In case of ties, the following tiebreakers will be applied in order:
- Goal difference (total points scored minus total points conceded)
- Head-to-head results (if exactly two models are tied)
- Average CPU steps per second during matches (higher is better)
- Model size (smaller is better)
- Training time (shorter is better)
- Overall implementation quality and creativity
- Model name coolness

The reviewers reserve the right to heavily penalize or even disqualify models that are too large (>> 25 MB) or that require an excessive amount of training resources (we should be able to replicate the training on a single GPU or on Google Colab in reasonable time) or that do not work properly out of the box and require excessive troubleshooting.
