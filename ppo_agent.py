import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        
        # Shared feature extractor / Policy Network (Actor)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim)
        )
        
        # Trainable log standard deviation for continuous Gaussian policy
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        # Value Network (Critic)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            # Sample action during exploration
            raw_action = probs.sample()
        else:
            raw_action = action
            
        action_logprob = probs.log_prob(raw_action).sum(1)
        entropy = probs.entropy().sum(1)
        
        # Actions are clamped to [-1.0, 1.0] for the environment
        clamped_action = torch.clamp(raw_action, -1.0, 1.0)
        
        return clamped_action, action_logprob, entropy, self.critic(x)

    def act_deterministic(self, x):
        """Used for deployment and ONNX export (no sampling, pure mean control)"""
        with torch.no_grad():
            action_mean = self.actor(x)
            return torch.clamp(action_mean, -1.0, 1.0)


class PPOTrainer:
    def __init__(
        self,
        env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu"
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.agent = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)

    def export_onnx(self, filepath="rocket_policy.onnx"):
        """Exports the trained Actor policy to ONNX format for running in browser Three.js"""
        self.agent.eval()
        dummy_input = torch.zeros(1, self.env.observation_space.shape[0], device=self.device)
        
        # Wrapper class to output only deterministic actions for ONNX
        class PolicyWrapper(nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor
            def forward(self, x):
                return torch.clamp(self.actor(x), -1.0, 1.0)

        wrapper = PolicyWrapper(self.agent.actor)
        torch.onnx.export(
            wrapper,
            dummy_input,
            filepath,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
            opset_version=14
        )
        print(f"[+] Policy successfully exported to {filepath}!")
