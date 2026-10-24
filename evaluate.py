import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from rocket_env import RocketLandingEnv
from ppo_agent import ActorCritic

def evaluate(model_path=None, episodes=5, render_plot=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(script_dir, "models", "best_rocket_model.pt")
    env = RocketLandingEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = ActorCritic(obs_dim, act_dim)
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"[+] Loaded weights from {model_path}")
    else:
        print(f"[!] Warning: {model_path} not found. Running with untrained policy.")

    agent.eval()

    all_trajectories = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0.0

        traj = {"x": [], "y": [], "vx": [], "vy": [], "theta": [], "thrust": [], "gimbal": []}

        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0)
            with torch.no_grad():
                action = agent.act_deterministic(obs_tensor).squeeze(0).numpy()

            raw_throttle = float(action[0])
            raw_gimbal = float(action[1])

            traj["x"].append(obs[0])
            traj["y"].append(obs[1])
            traj["vx"].append(obs[2])
            traj["vy"].append(obs[3])
            traj["theta"].append(np.degrees(obs[4]))
            traj["thrust"].append((raw_throttle + 1.0) / 2.0)
            traj["gimbal"].append(raw_gimbal * 15.0)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

        all_trajectories.append(traj)
        landed = info["altitude"] <= 0.0 and abs(info["vy"]) < 3.2 and abs(info["tilt_deg"]) < 9.0 and abs(info["dist_x"]) < 30.0
        vy_abs = abs(info["vy"])
        tilt_abs = abs(info["tilt_deg"])
        dist_abs = abs(info["dist_x"])
        if landed:
            if vy_abs < 1.6 and tilt_abs < 2.0 and dist_abs < 3.5:
                grade = "A+ (FLAWLESS)"
            elif vy_abs < 2.4 and tilt_abs < 4.5:
                grade = "A (EXCELLENT)"
            else:
                grade = "B (NOMINAL)"
        else:
            grade = "ANOMALY (CRASH)"

        print(f"Episode {ep + 1}: Grade: {grade:16s} | Steps: {step:3d} | Touchdown Vy: {info['vy']:5.2f} m/s | Tilt: {info['tilt_deg']:5.1f}° | Pad Error: {info['dist_x']:5.2f}m | Reward: {total_reward:6.1f}")

    if render_plot and len(all_trajectories) > 0:
        traj = all_trajectories[0]
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Rocket Autonomous Landing Trajectory", fontsize=14, fontweight="bold")

        # 1. 2D Flight Profile (X vs Y)
        axs[0, 0].plot(traj["x"], traj["y"], "b-", lw=2, label="Flight Path")
        axs[0, 0].axhline(0, color="gray", ls="--")
        axs[0, 0].plot(0, 0, "r^", ms=10, label="Landing Target (Pad)")
        axs[0, 0].set_xlabel("Horizontal Position x (m)")
        axs[0, 0].set_ylabel("Altitude y (m)")
        axs[0, 0].set_title("Descent Profile")
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()

        # 2. Velocities
        time_axis = np.arange(len(traj["y"])) * env.dt
        axs[0, 1].plot(time_axis, traj["vy"], "r-", label="Vertical Vy (m/s)")
        axs[0, 1].plot(time_axis, traj["vx"], "g--", label="Horizontal Vx (m/s)")
        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].set_ylabel("Velocity (m/s)")
        axs[0, 1].set_title("Linear Velocities")
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()

        # 3. Tilt Angle
        axs[1, 0].plot(time_axis, traj["theta"], "m-", lw=1.5, label="Tilt (deg)")
        axs[1, 0].axhline(0, color="k", ls=":")
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("Angle (degrees)")
        axs[1, 0].set_title("Attitude / Tilt Angle")
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend()

        # 4. Actuator Commands (TVC Gimbal & Throttle)
        axs[1, 1].plot(time_axis, traj["thrust"], "orange", label="Throttle (0-1)")
        axs[1, 1].plot(time_axis, traj["gimbal"], "c--", label="Gimbal Angle (deg)")
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("Actuator Value")
        axs[1, 1].set_title("Control Inputs (Thrust Vectoring)")
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].legend()

        plt.tight_layout()
        plot_path = os.path.join(script_dir, "models", "flight_telemetry.png")
        plt.savefig(plot_path, dpi=150)
        print(f"[+] Saved flight telemetry plot to {plot_path}")

if __name__ == "__main__":
    evaluate(episodes=5)
