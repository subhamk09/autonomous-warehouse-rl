# Autonomous Warehouse Slotting & Robot Scheduler using Reinforcement Learning

A simulation-first BTech AIML project for warehouse order fulfilment with multiple robots, grid navigation, battery-aware dispatch, rush orders, failures, and demand-based storage slotting.

## What is included
- `app.py`: interactive Streamlit dashboard with warehouse map, fleet status, order queue, simulation controls, disruption toggles, and performance charts.
- `warehouse_env.py`: Gymnasium environment with robot/order state, shortest-path navigation, battery use, task assignment actions, and shaped rewards.
- `train_ppo.py`: Stable-Baselines3 PPO training script; saves `models/warehouse_ppo.zip`.
- `requirements.txt`: Python dependencies.

**Important:** The dashboard currently uses a BFS + priority heuristic for interactive dispatch. PPO is trained separately using `train_ppo.py`; the app does not pretend that an untrained model is already active.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Train PPO
```bash
python train_ppo.py --timesteps 100000
```
The trained checkpoint is saved to `models/warehouse_ppo.zip`. Model files are git-ignored; keep large checkpoints in local storage or GitHub Releases if needed.

## Deploy the dashboard
1. Push the repository to GitHub.
2. Visit https://share.streamlit.io/
3. Choose this repository, branch `main`, and app file `app.py`.
4. Click Deploy.

## Metrics to report
Compare PPO with FIFO and nearest-robot baselines over identical random seeds: order throughput, average completion time, travel distance, battery consumption, and recovery after robot failure / aisle blockage.

## Project schedule (submission: 23 November 2026)
- 11–17 Sep: project and warehouse definition
- 18–24 Sep: simulation and path planning
- 25 Sep–8 Oct: RL environment and PPO training
- 9–23 Oct: advanced scheduling, slotting and disruptions
- 24 Oct–8 Nov: evaluation, dashboard and results
- 9–23 Nov: validation, report and final submission
