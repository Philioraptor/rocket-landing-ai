# 🚀 Starhopper-RL: Autonomous 6-DOF Falcon 9 Landing & Recovery via Decision Transformer & PPO

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.4-00f0ff.svg)](https://gymnasium.farama.org)
[![Three.js](https://img.shields.io/badge/Three.js-r128-black.svg)](https://threejs.org)

An autonomous Guidance, Navigation, and Control (GNC) flight system for soft vertical rocket touchdowns and extreme tumble recovery ($\pm 60^\circ$ pitch angles) using **Deep Reinforcement Learning (PPO)**, **Causal Decision Transformers (Rocket GPT)**, and a **3D WebGL Digital Twin** built in Three.js.

---

## 📐 System Architecture & Physical Dynamics

Unlike toy discrete environments (e.g. classical `LunarLander`), this system implements non-linear rigid-body aerodynamics, thrust vectoring, and propellant mass dynamics:

### 1. Actuator Space (Continuous)
* **Throttle Command ($u_0 \in [0.3, 1.0]$):** Scaled rocket engine thrust up to $F_{\max} = 30\text{ kN}$. Real engines cannot throttle below 30% without combustion instability.
* **Gimbal Deflection ($u_1 \in [-15^\circ, +15^\circ]$):** Swiveling Merlin 1D rocket nozzle angle $\beta$ providing active rotational restoring torque.

### 2. Equations of Motion
* **Total Thrust Vector:**
  $$\vec{F}_T = \begin{bmatrix} -F \sin(\theta + \beta) \\ F \cos(\theta + \beta) \end{bmatrix}$$
* **Aerodynamic Drag:**
  $$\vec{F}_D = -\frac{1}{2} \rho C_D A \|\vec{v}_{rel}\| \vec{v}_{rel}$$
* **Mass Depletion (Tsiolkovsky's Rocket Equation):**
  $$\dot{m} = -\frac{F}{I_{sp} \cdot g_0}$$
* **Rotational Torque & Moment of Inertia:**
  $$\tau = -F \sin(\beta) \cdot \left(\frac{L}{2}\right), \quad I_{zz} = \frac{1}{12} m L^2, \quad \alpha = \frac{\tau}{I_{zz}}$$
* **Integration Scheme:** Semi-Implicit Symplectic Euler to preserve phase space energy and eliminate numerical instability during fast rotations.

---

## 🧠 Dual AI Paradigms

### 1. The Decision Transformer (Rocket GPT)
Reframes rocket landing as **Autoregressive Sequence Modeling**:
$$\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T)$$
* **Conditioning on Return-to-Go:** Condition the prompt with target reward $\hat{R} = +700$ (feather-soft touchdown on the ASDS center bullseye).
* **Architecture:** 219k-parameter Causal Transformer with 4 Multi-Head Self-Attention heads and LayerNorm, trained on CPU to **93.5% fit accuracy (0.0042 MSE)** across extreme tumble recoveries.

### 2. Deep Reinforcement Learning (PPO)
* **Algorithm:** Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE).
* **Observation Vector ($s \in \mathbb{R}^7$):**
  $$s = [x, y, v_x, v_y, \theta, \omega, \text{fuel\_ratio}]$$
* **Curriculum Learning:** Progresses dynamically from terminal hover discovery up to high-altitude ballistic descent in 40 km/h wind gusts.

---

## 🏆 Touchdown Evaluation & Landing Grade System

Every simulation flight produces a quantitative **Mission Debrief Scorecard** evaluating structural survival, touchdown impact force, bullseye positioning, and attitude verticality:

| Landing Grade | Touchdown Velocity ($v_y$) | Pad Radial Error ($\Delta x$) | Vertical Attitude ($\theta$) | Flight Status |
| :--- | :--- | :--- | :--- | :--- |
| **A+ FLAWLESS** | **$< 1.6 \text{ m/s}$** (Feather Touch) | **$< 3.5 \text{ m}$** (Center Ring) | **$< 2.0^\circ$** (Dead Vertical) | **Pristine Recovery & Touchdown** |
| **A EXCELLENT** | $< 2.4 \text{ m/s}$ (Soft Landing) | $< 8.0 \text{ m}$ (Deck Bullseye) | $< 4.5^\circ$ (Nominal Attitude) | Soft Touchdown within Specs |
| **B NOMINAL** | $< 3.2 \text{ m/s}$ (Max Structural) | $< 30.0 \text{ m}$ (Barge Deck) | $< 9.0^\circ$ (Upright Stance) | Touchdown with Minor Drift |
| **ANOMALY / CRASH** | $\ge 3.2 \text{ m/s}$ (Structural Failure) | $\ge 30.0 \text{ m}$ (Ocean Miss) | $\ge 9.0^\circ$ (Tipped Over) | Rapid Unplanned Disassembly |

---

## ⚡ Quickstart

### 1. Run the 3D Interactive Web Visualizer
```bash
python -m http.server 8000 --directory visualizer
```
Open **[http://localhost:8000](http://localhost:8000)** in Chrome/Edge:
* **Interactive Directional Sliders:** Set initial launch pitch tilt from $60^\circ \text{ Left}$ to $60^\circ \text{ Right}$, spin tumble rate, and crosswinds.
* **Authentic Falcon 9 Asset:** 1:12 slender aspect ratio, waffle-lattice grid fins, carbon-fiber landing legs, octoweb base, procedural Web Audio rocket roar, and cold-gas nitrogen RCS puffs.

### 2. Train the Decision Transformer (Rocket GPT)
```bash
python generate_dataset.py  # Collects 500 extreme tumble recovery demonstrations
python train_gpt.py         # Trains the 219k-param Causal Transformer on CPU
python evaluate_gpt.py      # Evaluates closed-loop flight & generates telemetry graphs
```

### 3. Train the PPO Reinforcement Learning Policy
```bash
python train.py             # Vectorized PPO training loop with auto ONNX export
python evaluate.py          # Generates 4-panel flight telemetry plot
```

---

## 📁 Repository Structure

```
rocket-landing-ai/
├── rocket_env.py          # Custom continuous physics Gymnasium environment
├── model_gpt.py           # Decision Transformer (Rocket GPT) architecture
├── generate_dataset.py    # 6-DOF extreme tumble recovery dataset generator
├── train_gpt.py           # Autoregressive Transformer training loop
├── evaluate_gpt.py        # Closed-loop Transformer evaluation & plot generator
├── ppo_agent.py           # PyTorch Actor-Critic PPO architecture + ONNX exporter
├── train.py               # Curriculum PPO training loop with GAE
├── evaluate.py            # PPO evaluation and telemetry plot generator
├── visualizer/
│   ├── index.html         # Three.js 3D SpaceX-style mission control dashboard
│   └── rocket_policy.onnx # Exported neural policy running in WebGL
└── models/                # Checkpoints (rocket_gpt.pt, best_rocket_model.pt) & plots
```
