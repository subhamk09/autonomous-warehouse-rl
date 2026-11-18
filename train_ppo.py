"""Train PPO on the warehouse Gymnasium environment."""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from warehouse_env import WarehouseEnv

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--timesteps",type=int,default=100_000)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    env=WarehouseEnv(seed=args.seed)
    check_env(env,warn=True)
    model=PPO("MlpPolicy",env,verbose=1,seed=args.seed,learning_rate=3e-4,
              n_steps=1024,batch_size=64,gamma=0.99)
    model.learn(total_timesteps=args.timesteps)
    Path("models").mkdir(exist_ok=True)
    model.save("models/warehouse_ppo")
    print("Saved models/warehouse_ppo.zip")

if __name__=="__main__":
    main()
