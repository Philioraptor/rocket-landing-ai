import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from rocket_env import RocketLandingEnv
from model_gpt import RocketDecisionTransformer

def evaluate_gpt(episodes=5, target_return=700.0, render_plot=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "rocket_gpt.pt")

    device = torch.device("cpu")
    model = RocketDecisionTransformer(
        state_dim=7,
        act_dim=2,
        d_model=64,
        nhead=4,
        num_layers=3,
        context_len=20
    ).to(device)

    if not os.path.exists(model_path):
        print(f"[!] Model {model_path} not found.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[+] Loaded Rocket GPT weights from {model_path}!")
    print(f"=== Conditioning Rocket GPT with Target Return: +{target_return:.0f} (Pristine Landing) ===")

    env = RocketLandingEnv()
    context_len = 20
    all_trajectories = []

    for ep in range(episodes):
        env.difficulty = 0.5  # Test on realistic descent
        obs, _ = env.reset()

        # Initialize sequence buffers for Transformer context
        states_seq = [obs]
        actions_seq = [np.zeros(2, dtype=np.float32)]
        rtg_seq = [np.array([target_return], dtype=np.float32)]
        timesteps_seq = [0]

        done = False
        step = 0
        total_reward = 0.0

        traj = {"x": [], "y": [], "vx": [], "vy": [], "theta": [], "throttle": [], "gimbal": []}

        while not done:
            # Prepare context window of length min(step+1, context_len)
            k = min(len(states_seq), context_len)
            s_tensor = torch.tensor(states_seq[-k:], dtype=torch.float32).unsqueeze(0).to(device)
            a_tensor = torch.tensor(actions_seq[-k:], dtype=torch.float32).unsqueeze(0).to(device)
            r_tensor = torch.tensor(rtg_seq[-k:], dtype=torch.float32).unsqueeze(0).to(device)
            t_tensor = torch.tensor(timesteps_seq[-k:], dtype=torch.long).unsqueeze(0).to(device)

            # Query the Transformer for the next flight action
            action_pred = model.get_action(s_tensor, a_tensor, r_tensor, t_tensor)
            action = action_pred.squeeze(0).cpu().numpy()

            # Record trajectory
            traj["x"].append(obs[0])
            traj["y"].append(obs[1])
            traj["vx"].append(obs[2])
            traj["vy"].append(obs[3])
            traj["theta"].append(np.degrees(obs[4]))
            traj["throttle"].append((action[0] + 1.0) / 2.0)
            traj["gimbal"].append(action[1] * 15.0)

            # Step physical environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Update sequence for next autoregressive step
            next_rtg = max(0.0, rtg_seq[-1][0] - reward)
            states_seq.append(next_obs)
            actions_seq.append(action)
            rtg_seq.append(np.array([next_rtg], dtype=np.float32))
            timesteps_seq.append(step + 1)

            obs = next_obs
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

        print(f"Flight #{ep+1}: Grade: {grade:16s} | Steps: {step:3d} | Touchdown Vy: {info['vy']:5.2f} m/s | Tilt: {info['tilt_deg']:5.1f}° | Pad Error: {info['dist_x']:5.2f}m | Reward: {total_reward:6.1f}")

    if render_plot and len(all_trajectories) > 0:
        traj = all_trajectories[0]
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Rocket GPT (Decision Transformer) Autonomous Landing Profile", fontsize=13, fontweight="bold")

        axs[0, 0].plot(traj["x"], traj["y"], "c-", lw=2, label="Transformer Trajectory")
        axs[0, 0].axhline(0, color="gray", ls="--")
        axs[0, 0].plot(0, 0, "r^", ms=10, label="Target Pad")
        axs[0, 0].set_title("Descent Profile (X vs Y)")
        axs[0, 0].set_xlabel("Horizontal Error (m)")
        axs[0, 0].set_ylabel("Altitude (m)")
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()

        time_axis = np.arange(len(traj["y"])) * env.dt
        axs[0, 1].plot(time_axis, traj["vy"], "r-", label="Vertical Vy (m/s)")
        axs[0, 1].plot(time_axis, traj["vx"], "g--", label="Horizontal Vx (m/s)")
        axs[0, 1].set_title("Velocities Over Time")
        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].set_ylabel("Velocity (m/s)")
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()

        axs[1, 0].plot(time_axis, traj["theta"], "m-", label="Tilt Angle (deg)")
        axs[1, 0].axhline(0, color="k", ls=":")
        axs[1, 0].set_title("Attitude / Tilt Angle")
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("Degrees")
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend()

        axs[1, 1].plot(time_axis, traj["throttle"], "orange", label="Throttle")
        axs[1, 1].plot(time_axis, traj["gimbal"], "c--", label="Gimbal (deg)")
        axs[1, 1].set_title("Transformer Control Output (TVC)")
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("Actuator Value")
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].legend()

        plt.tight_layout()
        plot_path = os.path.join(script_dir, "models", "rocket_gpt_telemetry.png")
        plt.savefig(plot_path, dpi=150)
        print(f"[+] Saved Transformer flight telemetry to {plot_path}")

if __name__ == "__main__":
    evaluate_gpt()
