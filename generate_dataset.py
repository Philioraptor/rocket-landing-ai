import os
import pickle
import numpy as np
from rocket_env import RocketLandingEnv

def collect_recovery_dataset(num_episodes=500, save_path="data/rocket_trajectories.pkl"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_save_path = os.path.join(script_dir, save_path)
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

    env = RocketLandingEnv()
    env.max_steps = 1000
    trajectories = []
    success_count = 0

    print(f"=== Generating {num_episodes} Extreme Tumble Recovery Demonstrations for Rocket GPT ===")

    for ep in range(num_episodes):
        diff = float(np.random.uniform(0.2, 1.0))
        env.difficulty = diff
        obs, _ = env.reset()

        # Intentionally inject extreme initial tilt & spin on 50% of the flights (Recovery curriculum)
        if np.random.rand() > 0.4:
            initial_tilt = np.random.uniform(-np.radians(35.0), np.radians(35.0))
            initial_omega = np.random.uniform(-0.35, 0.35)
            obs[4] = initial_tilt
            obs[5] = initial_omega

        obs_list = []
        action_list = []
        reward_list = []

        done = False
        step = 0

        while not done:
            x, y, vx, vy, theta, omega, fuel = obs

            # High-Gain Spin Arrest & Attitude Hold
            target_theta = float(x * 0.015 + vx * 0.035)
            target_theta = np.clip(target_theta, -0.25, 0.25)
            
            # Gimbal torque calculation with derivative spin damping
            angle_error = theta - target_theta
            gim = 1.0 * (angle_error * 2.8 + omega * 2.4)

            # If tumbling wildly, fire engine to give gimbal aerodynamic torque authority
            is_tumbling = abs(theta) > 0.25 or abs(omega) > 0.15

            if is_tumbling:
                th_cmd = 0.55
            elif y < 15.0:
                target_vy = -1.3
                a_needed = (target_vy - vy) * 8.0
                th_cmd = min(1.0, max(0.0, (env.mass * (env.g + a_needed)) / env.max_thrust))
            else:
                target_vy = -max(2.0, min(25.0, (y / 20.0) * 8.0))
                a_needed = (target_vy - vy) * 3.5
                th_cmd = min(1.0, max(0.0, (env.mass * (env.g + a_needed)) / env.max_thrust))

            raw_th = float(th_cmd * 2.0 - 1.0)
            raw_gim = float(np.clip(gim / env.max_gimbal, -1.0, 1.0))

            action = np.array([raw_th, raw_gim], dtype=np.float32)

            obs_list.append(obs)
            action_list.append(action)

            next_obs, reward, terminated, truncated, info = env.step(action)
            reward_list.append(reward)

            obs = next_obs
            step += 1
            done = terminated or truncated

        rewards = np.array(reward_list, dtype=np.float32)
        returns_to_go = np.zeros_like(rewards)
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return += rewards[t]
            returns_to_go[t] = running_return

        is_success = rewards[-1] > 200.0
        if is_success:
            success_count += 1

        trajectories.append({
            "observations": np.array(obs_list, dtype=np.float32),
            "actions": np.array(action_list, dtype=np.float32),
            "rewards": rewards,
            "returns_to_go": returns_to_go,
            "timesteps": np.arange(len(obs_list), dtype=np.int64),
            "total_return": running_return,
            "success": is_success
        })

    with open(full_save_path, "wb") as f:
        pickle.dump(trajectories, f)

    success_rate = (success_count / num_episodes) * 100.0
    print(f"[+] Dataset saved to {full_save_path}!")
    print(f"[+] Total Demonstrations: {len(trajectories)} | Successful Recoveries & Touchdowns: {success_count}/{num_episodes} ({success_rate:.1f}%)")

if __name__ == "__main__":
    collect_recovery_dataset()
