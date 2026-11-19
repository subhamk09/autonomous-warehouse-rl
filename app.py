
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
import random

st.set_page_config(
    page_title="Autonomous Warehouse RL",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #e5e7eb;
}
[data-testid="stMetric"] {
    background: #111c30;
    padding: 14px;
    border: 1px solid #263650;
    border-radius: 12px;
}
.block-container {
    padding-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Autonomous Warehouse Slotting & Robot Scheduler")
st.caption(
    "Simulation dashboard • Grid warehouse • Dispatch • "
    "Dynamic slotting • PPO-ready environment"
)
st.info(
    "This is a simulated warehouse. Until a trained PPO model is loaded, "
    "dispatch uses a transparent shortest-path heuristic—not a trained RL policy."
)

if "sim" not in st.session_state:
    st.session_state.sim = None


# BFS shortest-path algorithm
def bfs(grid, start, goal):
    q = deque([(start, [])])
    seen = {start}

    while q:
        (r, c), path = q.popleft()

        if (r, c) == goal:
            return path + [(r, c)]

        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            p = (r + dr, c + dc)

            if (
                0 <= p[0] < grid.shape[0]
                and 0 <= p[1] < grid.shape[1]
                and grid[p] != 1
                and p not in seen
            ):
                seen.add(p)
                q.append((p, path + [(r, c)]))

    return []


# Create a new warehouse simulation
def new_sim(rows, cols, nrobots, norders, seed):
    rng = np.random.default_rng(seed)

    grid = np.zeros((rows, cols), dtype=int)

    for _ in range(max(2, rows * cols // 14)):
        p = (
            int(rng.integers(rows)),
            int(rng.integers(cols))
        )

        if p not in [(0, 0), (rows - 1, cols - 1)]:
            grid[p] = 1

    free = list(zip(*np.where(grid == 0)))
    rng.shuffle(free)

    robots = []

    for i in range(min(nrobots, len(free))):
        p = free.pop()

        robots.append({
            "id": f"R{i+1}",
            "pos": p,
            "battery": int(rng.integers(55, 101)),
            "busy": False,
            "failed": False,
            "path": [],
            "done": 0
        })

    products = [
        f"SKU-{i+1:02d}"
        for i in range(max(6, norders))
    ]

    orders = []

    for i in range(norders):
        p = free.pop() if free else (rows - 1, cols - 1)

        orders.append({
            "id": f"O{i+1:03d}",
            "sku": products[i % len(products)],
            "pick": p,
            "priority": int(rng.choice([1, 1, 1, 2, 3])),
            "status": "Pending"
        })

    return {
        "grid": grid,
        "robots": robots,
        "orders": orders,
        "tick": 0,
        "distance": 0,
        "completed": 0,
        "history": []
    }


# Assign orders to available robots
def dispatch(sim):
    pending = sorted(
        [o for o in sim["orders"] if o["status"] == "Pending"],
        key=lambda x: -x["priority"]
    )

    for o in pending:
        candidates = []

        for rb in sim["robots"]:
            if not rb["busy"] and not rb["failed"] and rb["battery"] > 8:
                p = bfs(sim["grid"], rb["pos"], o["pick"])

                if p:
                    candidates.append((len(p), rb, p))

        if candidates:
            _, rb, p = min(candidates, key=lambda x: x[0])

            rb["path"] = p[1:]
            rb["busy"] = True
            o["status"] = "Assigned"
            o["robot"] = rb["id"]


# Move robots and update simulation
def step(sim):
    sim["tick"] += 1

    for rb in sim["robots"]:
        if rb["busy"] and not rb["failed"]:

            if rb["path"]:
                rb["pos"] = rb["path"].pop(0)
                rb["battery"] = max(0, rb["battery"] - 1)
                sim["distance"] += 1

            if not rb["path"]:
                rb["busy"] = False
                rb["done"] += 1

                for o in sim["orders"]:
                    if (
                        o["status"] == "Assigned"
                        and o.get("robot") == rb["id"]
                    ):
                        o["status"] = "Completed"
                        sim["completed"] += 1
                        break

        elif not rb["failed"] and rb["battery"] < 25:
            rb["battery"] = min(100, rb["battery"] + 5)

    sim["history"].append({
        "Step": sim["tick"],
        "Completed": sim["completed"],
        "Distance": sim["distance"]
    })


# Sidebar controls
with st.sidebar:
    st.header("Simulation controls")

    rows = st.slider("Warehouse rows", 6, 16, 10)
    cols = st.slider("Warehouse columns", 6, 16, 12)
    nr = st.slider("Number of robots", 1, 6, 3)
    no = st.slider("Number of orders", 3, 20, 10)
    seed = st.number_input("Random seed", 0, 9999, 42)

    if st.button("Create / reset warehouse", use_container_width=True):
        st.session_state.sim = new_sim(
            rows, cols, nr, no, int(seed)
        )

    st.divider()
    st.caption("Disruption controls")

    if st.session_state.sim is not None:
        ids = [
            r["id"]
            for r in st.session_state.sim["robots"]
        ]

        chosen = st.selectbox("Robot", ids)

        if st.button("Toggle robot failure"):
            rb = next(
                r for r in st.session_state.sim["robots"]
                if r["id"] == chosen
            )

            rb["failed"] = not rb["failed"]
            rb["busy"] = False

        if st.button("Add rush order"):
            sim = st.session_state.sim

            free = list(zip(*np.where(sim["grid"] == 0)))
            p = random.choice(free)
            oid = f"O{len(sim['orders'])+1:03d}"

            sim["orders"].append({
                "id": oid,
                "sku": "RUSH-SKU",
                "pick": p,
                "priority": 3,
                "status": "Pending"
            })

        if st.button("Dispatch pending orders", use_container_width=True):
            dispatch(st.session_state.sim)

        if st.button("Run 1 simulation step", use_container_width=True):
            step(st.session_state.sim)

        if st.button("Run 10 steps", use_container_width=True):
            for _ in range(10):
                dispatch(st.session_state.sim)
                step(st.session_state.sim)


# Initialize simulation
if st.session_state.sim is None:
    st.session_state.sim = new_sim(10, 12, 3, 10, 42)

sim = st.session_state.sim

pending = sum(
    o["status"] == "Pending"
    for o in sim["orders"]
)

completed = sim["completed"]


# Dashboard metrics
c1, c2, c3, c4 = st.columns(4)

c1.metric("Simulation step", sim["tick"])
c2.metric("Orders completed", completed)
c3.metric("Orders pending", pending)
c4.metric("Total travel cells", sim["distance"])


# Warehouse map and robot information
left, right = st.columns([1.3, 1])

with left:
    st.subheader("Warehouse map")

    view = np.zeros(
        (*sim["grid"].shape, 3),
        dtype=float
    ) + 0.12

    # Red: blocked areas
    for r, c in zip(*np.where(sim["grid"] == 1)):
        view[r, c] = [0.85, 0.22, 0.22]

    # Yellow/orange: order locations
    for o in sim["orders"]:
        if o["status"] != "Completed":
            view[o["pick"]] = (
                [0.95, 0.68, 0.15]
                if o["priority"] < 3
                else [1, 0.35, 0.2]
            )

    # Cyan: normal robots
    # White: failed robots
    for i, rb in enumerate(sim["robots"]):
        view[rb["pos"]] = (
            [0.15, 0.75, 0.95]
            if not rb["failed"]
            else [1.0, 1.0, 1.0]
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.imshow(view, interpolation="nearest")

    ax.set_xticks(range(sim["grid"].shape[1]))
    ax.set_yticks(range(sim["grid"].shape[0]))

    ax.grid(color="#64748b", alpha=0.35)

    ax.set_title(
        "Red: blocked • Yellow: order • "
        "Cyan: robot • White: failed"
    )

    st.pyplot(fig)
    plt.close(fig)


with right:
    st.subheader("Robot fleet")

    st.dataframe(
        pd.DataFrame([
            {
                "Robot": r["id"],
                "Battery %": r["battery"],
                "State": (
                    "Failed" if r["failed"]
                    else ("Moving" if r["busy"] else "Idle")
                ),
                "Completed": r["done"],
                "Position": str(r["pos"])
            }
            for r in sim["robots"]
        ]),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Order queue")

    st.dataframe(
        pd.DataFrame([
            {
                "Order": o["id"],
                "SKU": o["sku"],
                "Priority": o["priority"],
                "Status": o["status"]
            }
            for o in sim["orders"]
        ]),
        use_container_width=True,
        hide_index=True
    )


# Performance graph
st.subheader("Performance")

if sim["history"]:
    hist = (
        pd.DataFrame(sim["history"])
        .drop_duplicates("Step", keep="last")
        .set_index("Step")
    )

    st.line_chart(hist)

else:
    st.caption(
        "Run simulation steps to populate performance charts."
    )


# Dynamic slotting recommendation
st.subheader("Dynamic slotting recommendation")

sku_counts = {}

for o in sim["orders"]:
    sku_counts[o["sku"]] = (
        sku_counts.get(o["sku"], 0)
        + (2 if o["priority"] == 3 else 1)
    )

slots = sorted(
    sku_counts.items(),
    key=lambda x: -x[1]
)

st.dataframe(
    pd.DataFrame([
        {
            "Suggested pick zone": f"Zone {i+1}",
            "SKU": sku,
            "Demand weight": w,
            "Reason": "Higher-demand / priority items closer to dispatch"
        }
        for i, (sku, w) in enumerate(slots)
    ]),
    use_container_width=True,
    hide_index=True
)


# RL model status
st.subheader("RL model status")

model_path = Path("models/warehouse_ppo.zip")

if model_path.exists():
    st.success(
        "PPO model file found at models/warehouse_ppo.zip. "
        "Train/evaluate it using train_ppo.py and connect its "
        "policy to dispatch for learned actions."
    )

else:
    st.warning(
        "No trained PPO checkpoint detected. Current scheduler "
        "uses BFS shortest-path + priority dispatch."
    )


st.caption(
    "Project target submission: 23 November 2026 • "
    "All operations shown here are synthetic simulation data."
)
