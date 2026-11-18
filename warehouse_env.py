"""Gymnasium environment for a compact multi-robot warehouse dispatch task."""
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class WarehouseEnv(gym.Env):
    """Discrete task-selection environment.

    Action: choose an order index, or choose the final WAIT action.
    The closest available robot is assigned to the selected pending order.
    One environment step advances that robot along its BFS path.
    """
    metadata = {"render_modes": []}

    def __init__(self, rows=8, cols=8, robots=2, orders=8, max_steps=300, seed=None):
        super().__init__()
        self.rows, self.cols, self.n_robots, self.n_orders = rows, cols, robots, orders
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(orders + 1)
        # robot row/col/battery/busy for each robot; order row/col/status/priority for each order; time
        self.observation_space = spaces.Box(0.0, 1.0, shape=(robots*4 + orders*4 + 1,), dtype=np.float32)
        self._seed = seed
        self.reset(seed=seed)

    def _path(self, start, goal):
        q=deque([(start, [])]); seen={start}
        while q:
            pos,path=q.popleft()
            if pos==goal: return path+[pos]
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nxt=(pos[0]+dr,pos[1]+dc)
                if 0<=nxt[0]<self.rows and 0<=nxt[1]<self.cols and nxt not in self.blocked and nxt not in seen:
                    seen.add(nxt); q.append((nxt,path+[pos]))
        return []

    def _obs(self):
        vals=[]
        for r in self.robots:
            vals += [r["pos"][0]/(self.rows-1),r["pos"][1]/(self.cols-1),r["battery"]/100,float(r["busy"])]
        for o in self.orders:
            vals += [o["pos"][0]/(self.rows-1),o["pos"][1]/(self.cols-1),o["status"]/2,o["priority"]/3]
        vals.append(self.steps/self.max_steps)
        return np.asarray(vals,dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps=0; self.completed=0; self.travel=0
        self.blocked=set()
        self.robots=[{"pos":(0,i),"battery":100,"path":[],"busy":False} for i in range(self.n_robots)]
        self.orders=[]
        for i in range(self.n_orders):
            self.orders.append({"pos":(int(self.np_random.integers(1,self.rows)),int(self.np_random.integers(self.cols))),
                                "status":0,"priority":int(self.np_random.integers(1,4))})
        return self._obs(), {"completed":self.completed}

    def step(self, action):
        reward=-0.02
        action=int(action)
        if action < self.n_orders:
            order=self.orders[action]
            if order["status"]==0:
                candidates=[]
                for idx,robot in enumerate(self.robots):
                    if not robot["busy"] and robot["battery"]>5:
                        path=self._path(robot["pos"],order["pos"])
                        if path: candidates.append((len(path),idx,path))
                if candidates:
                    _,idx,path=min(candidates,key=lambda x:x[0])
                    self.robots[idx]["path"]=path[1:]
                    self.robots[idx]["busy"]=True
                    order["status"]=1
                    order["robot"]=idx
                    reward += 0.15*order["priority"]
                else: reward -= 0.25
            else: reward -= 0.15
        else:
            reward -= 0.03

        for robot in self.robots:
            if robot["busy"] and robot["path"]:
                robot["pos"]=robot["path"].pop(0)
                robot["battery"]=max(0,robot["battery"]-1)
                self.travel+=1
                reward-=0.01
                if not robot["path"]:
                    robot["busy"]=False
                    robot["battery"]=max(20,robot["battery"])
                    for order in self.orders:
                        if order["status"]==1 and order.get("robot")==self.robots.index(robot):
                            order["status"]=2; self.completed+=1
                            reward += 2.0 + 0.5*order["priority"]
                            break
        self.steps+=1
        terminated=self.completed==self.n_orders
        truncated=self.steps>=self.max_steps
        if terminated: reward+=5.0
        return self._obs(),float(reward),terminated,truncated,{"completed":self.completed,"travel_distance":self.travel}

    def render(self):
        return {"robots":self.robots,"orders":self.orders,"completed":self.completed}
