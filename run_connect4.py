# run_connect4.py
import random
import numpy as np

# EDIT to your file/class
from connect4_env import GymnasiumConnectFour as ConnectFourEnv

def pick_legal_action(info, env):
    # Prefer using an action_mask when provided
    mask = info.get("action_mask", None)
    if mask is not None:
        legal = np.flatnonzero(np.asarray(mask))
        return int(random.choice(legal))
    # Fallback: sample until the env accepts (avoid if your env penalizes/terminates on illegal)
    for _ in range(100):
        a = env.action_space.sample()
        return a
    raise RuntimeError("Could not pick an action")

def main():
    env = ConnectFourEnv()
    obs, info = env.reset()
    print("obs shape:", getattr(obs, "shape", type(obs)))
    print("action space:", env.action_space)
    print("initial legal actions:", int(np.sum(info.get("action_mask", []))) if "action_mask" in info else "n/a")
    if hasattr(env, "render"):
        print("\nInitial board:")
        env.render()

    total_reward = 0.0
    terminated = truncated = False
    steps = 0
    max_steps = 2000

    while not (terminated or truncated) and steps < max_steps:
        action = pick_legal_action(info, env)
        obs, reward, terminated, truncated, info, action = env.step(action)
        total_reward += float(reward)
        steps += 1
        # Print every few steps
        if steps % 5 == 0 and hasattr(env, "render"):
            print(f"\nAfter {steps} steps (reward so far {total_reward}):")
            env.render()

    print(f"\nEpisode finished. steps={steps}, total_reward={total_reward}, terminated={terminated}, truncated={truncated}, reason={info.get('reason')}, last_move={action}")
    if terminated:
        print("Final board:")
        env.render()

if __name__ == "__main__":
    main()