import os
import sys
import time
import numpy as np
import torch

from rocket_env import RocketLandingEnv
from ppo_agent import PPOTrainer

import shutil

def train(
    total_timesteps=300000,
    num_steps=2048,
    num_minibatches=32,
    update_epochs=10,
    save_dir="models",
    resume=True
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, save_dir)
    visualizer_dir = os.path.join(script_dir, "visualizer")
    os.makedirs(save_path, exist_ok=True)

    env = RocketLandingEnv()
    trainer = PPOTrainer(env, learning_rate=3e-4, device="cpu")

    weights_file = os.path.join(save_path, "best_rocket_model.pt")
    if resume and os.path.exists(weights_file):
        try:
            trainer.agent.load_state_dict(torch.load(weights_file, map_location="cpu"))
            print(f"[+] Loaded existing weights from {weights_file} to continue training!")
        except Exception as e:
            print(f"[!] Could not load previous weights: {e}")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    batch_size = num_steps
    minibatch_size = batch_size // num_minibatches

    # Storage buffers
    obs_buf = torch.zeros((num_steps, obs_dim))
    actions_buf = torch.zeros((num_steps, act_dim))
    logprobs_buf = torch.zeros(num_steps)
    rewards_buf = torch.zeros(num_steps)
    dones_buf = torch.zeros(num_steps)
    values_buf = torch.zeros(num_steps)

    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs)
    next_done = torch.zeros(1)

    num_updates = total_timesteps // batch_size
    print(f"=== Starting Rocket PPO Training on CPU ===")
    print(f"Total Steps: {total_timesteps} | Batch Size: {batch_size} | Updates: {num_updates}")
    print("-" * 65)

    recent_returns = []
    success_count = 0
    total_landings_attempted = 0
    best_avg_return = -float("inf")

    current_ep_return = 0.0

    for update in range(1, num_updates + 1):
        # Curriculum Update: Smoothly transition from easy hover to high-altitude storm
        env.difficulty = min(1.0, global_step / 150000.0)

        # 1. Rollout Phase: Collect transitions using current policy
        trainer.agent.eval()
        for step in range(num_steps):
            global_step += 1
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = trainer.agent.get_action_and_value(next_obs.unsqueeze(0))
                values_buf[step] = value.flatten()
            actions_buf[step] = action.flatten()
            logprobs_buf[step] = logprob.flatten()

            # Step physical environment
            action_np = action.squeeze(0).cpu().numpy()
            step_obs, reward, terminated, truncated, info = env.step(action_np)
            
            rewards_buf[step] = float(reward)
            current_ep_return += float(reward)
            done = terminated or truncated
            
            if done:
                total_landings_attempted += 1
                recent_returns.append(current_ep_return)
                if reward > 300.0: # Landed successfully criteria
                    success_count += 1
                current_ep_return = 0.0
                next_obs, _ = env.reset()
                next_done = torch.tensor(1.0)
            else:
                next_done = torch.tensor(0.0)
            
            next_obs = torch.Tensor(step_obs if not done else next_obs)

        # 2. Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = trainer.agent.get_value(next_obs.unsqueeze(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + trainer.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + trainer.gamma * trainer.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # Flatten batch
        b_obs = obs_buf
        b_logprobs = logprobs_buf
        b_actions = actions_buf
        b_advantages = advantages
        b_returns = returns

        # 3. Optimizing the Policy and Value Network
        trainer.agent.train()
        b_inds = np.arange(batch_size)
        clipfracs = []

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = trainer.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs += [((ratio - 1.0).abs() > trainer.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                # Advantage normalization
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (Clipped surrogate)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - trainer.clip_coef, 1 + trainer.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss (Exploration encouragement)
                entropy_loss = entropy.mean()
                
                loss = pg_loss - trainer.ent_coef * entropy_loss + v_loss * trainer.vf_coef

                trainer.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.agent.parameters(), trainer.max_grad_norm)
                trainer.optimizer.step()

        # Logging stats
        elapsed = time.time() - start_time
        sps = int(global_step / elapsed)
        avg_return = np.mean(recent_returns[-20:]) if len(recent_returns) > 0 else 0.0
        success_rate = (success_count / max(1, total_landings_attempted)) * 100.0

        print(f"Update {update:3d}/{num_updates} | Step: {global_step:7d} | Diff: {env.difficulty:4.2f} | SPS: {sps} | "
              f"Avg Return: {avg_return:7.1f} | Landings: {success_count}/{total_landings_attempted} ({success_rate:4.1f}%)")
        sys.stdout.flush()

        # Save checkpoint if beating best
        if avg_return > best_avg_return and len(recent_returns) >= 10:
            best_avg_return = avg_return
            torch.save(trainer.agent.state_dict(), os.path.join(save_path, "best_rocket_model.pt"))
            onnx_path = os.path.join(save_path, "rocket_policy.onnx")
            trainer.export_onnx(onnx_path)
            # Also sync to visualizer
            try:
                shutil.copy2(onnx_path, os.path.join(visualizer_dir, "rocket_policy.onnx"))
            except Exception:
                pass

    print("\n[+] Training Complete!")
    # Always export final onnx
    onnx_path = os.path.join(save_path, "rocket_policy.onnx")
    trainer.export_onnx(onnx_path)
    try:
        shutil.copy2(onnx_path, os.path.join(visualizer_dir, "rocket_policy.onnx"))
    except Exception:
        pass

if __name__ == "__main__":
    train(total_timesteps=200000)
