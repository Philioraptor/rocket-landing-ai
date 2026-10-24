import torch
import torch.nn as nn
import numpy as np

class RocketDecisionTransformer(nn.Module):
    """
    Decision Transformer (Rocket GPT) for Autonomous GNC & Landing.
    
    Reframes Reinforcement Learning as Autoregressive Sequence Modeling.
    Treats flight trajectories as sequences of (Return-to-Go, State, Action) tokens.
    """
    def __init__(
        self,
        state_dim=7,
        act_dim=2,
        d_model=64,
        nhead=4,
        num_layers=3,
        max_ep_len=1000,
        context_len=20,
        dropout=0.1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.context_len = context_len

        # Linear projection embeddings for each modality
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Linear(act_dim, d_model)
        self.embed_timestep = nn.Embedding(max_ep_len, d_model)
        self.embed_ln = nn.LayerNorm(d_model)

        # Causal Multi-Head Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action prediction head from state token representation
        self.predict_action = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, act_dim),
            nn.Tanh() # Clamp actions to [-1.0, 1.0]
        )

    def forward(self, states, actions, returns_to_go, timesteps):
        """
        states: (B, K, state_dim)
        actions: (B, K, act_dim)
        returns_to_go: (B, K, 1)
        timesteps: (B, K)
        """
        batch_size, seq_len = states.shape[0], states.shape[1]

        time_emb = self.embed_timestep(timesteps)

        # Embed each modality and add positional timestep embedding
        rtg_emb = self.embed_rtg(returns_to_go) + time_emb
        state_emb = self.embed_state(states) + time_emb
        act_emb = self.embed_action(actions) + time_emb

        # Interleave tokens into sequence: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # Shape: (B, 3 * K, d_model)
        stacked_tokens = torch.stack((rtg_emb, state_emb, act_emb), dim=2)
        stacked_tokens = stacked_tokens.reshape(batch_size, 3 * seq_len, self.d_model)
        stacked_tokens = self.embed_ln(stacked_tokens)

        # Causal triangular attention mask (prevent attending to future tokens)
        total_tokens = 3 * seq_len
        causal_mask = nn.Transformer.generate_square_subsequent_mask(total_tokens).to(states.device)

        # Pass through Transformer
        hidden_states = self.transformer(stacked_tokens, mask=causal_mask, is_causal=True)

        # Extract hidden states at the `state` token positions (indices 1, 4, 7, ... -> 3*i + 1)
        state_tokens = hidden_states[:, 1::3, :]

        # Predict continuous actions
        action_preds = self.predict_action(state_tokens)
        return action_preds

    def get_action(self, states, actions, returns_to_go, timesteps):
        """
        Inference call during flight simulation.
        Predicts the action for the most recent timestep.
        """
        self.eval()
        with torch.no_grad():
            action_preds = self.forward(states, actions, returns_to_go, timesteps)
            # Return action prediction for the latest state in the context window
            return action_preds[:, -1, :]
