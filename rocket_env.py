import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RocketLandingEnv(gym.Env):
    """
    Continuous 2D/Planar 3-DOF Rocket Landing Environment with Thrust Vector Control (TVC).
    
    Coordinate frame:
      - x: Horizontal distance to landing pad (meters, target is x = 0)
      - y: Altitude above ground (meters, target touchdown is y = 0)
      - theta: Rocket tilt angle from vertical (radians, 0 = perfectly upright, + = tilted right)
      - vx, vy: Linear velocity (m/s)
      - omega: Angular velocity (rad/s)
      - m: Rocket mass (dry mass + fuel) (kg)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Rocket Physical Parameters (Falcon 9 / Starhopper scaled)
        self.dry_mass = 1000.0          # kg
        self.fuel_mass_initial = 500.0   # kg
        self.length = 10.0               # meters (height of rocket)
        self.radius = 1.0                # meters
        self.g = 9.81                    # m/s^2 (gravity)
        self.max_thrust = 25000.0        # Newtons (~25 kN, TWR ~ 1.7 at full fuel)
        self.min_thrust_ratio = 0.3      # Minimum engine throttle limit (cannot throttle below 30% when ignited)
        self.max_gimbal = np.radians(15.0) # +/- 15 degrees max gimbal nozzle deflection
        self.isp = 300.0                 # Specific impulse in seconds
        self.dt = 0.05                   # Simulation step time: 20 Hz control loop
        self.max_steps = 600             # 30 seconds max episode duration

        # State Space: [x, y, vx, vy, theta, omega, fuel_ratio]
        # Normalized bounds for neural network observation stability
        high = np.array([
            500.0,   # x (m)
            1000.0,  # y (m)
            100.0,   # vx (m/s)
            100.0,   # vy (m/s)
            np.pi,   # theta (rad)
            10.0,    # omega (rad/s)
            1.0      # fuel_ratio (0 to 1)
        ], dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.difficulty = 0.0  # Curriculum level from 0.0 (terminal hover/land) to 1.0 (full high-altitude storm)

        # Action Space: Continuous [throttle, gimbal]
        # action[0]: throttle in [-1, 1] mapped to [min_thrust_ratio, 1.0] or 0 (if cut off)
        # action[1]: gimbal angle in [-1, 1] mapped to [-max_gimbal, +max_gimbal]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.state = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Curriculum Learning: Scale starting altitude & velocity with difficulty
        d = float(np.clip(self.difficulty, 0.0, 1.0))

        # Easy (d=0): starts at 40-70m, low horizontal error, gentle descent
        # Hard (d=1): starts at 250-350m, wild horizontal spread, high descent speed
        y_min = 40.0 + d * 210.0
        y_max = 70.0 + d * 280.0
        y0 = float(self.np_random.uniform(y_min, y_max))

        x_spread = 15.0 + d * 55.0
        x0 = float(self.np_random.uniform(-x_spread, x_spread))

        vx_max = 3.0 + d * 15.0
        vx0 = float(self.np_random.uniform(-vx_max, vx_max))

        vy_min = -12.0 - d * 18.0
        vy_max = -5.0 - d * 15.0
        vy0 = float(self.np_random.uniform(vy_min, vy_max))

        tilt_max = np.radians(6.0 + d * 16.0)
        theta0 = float(self.np_random.uniform(-tilt_max, tilt_max))
        omega0 = float(self.np_random.uniform(-0.05 - d * 0.08, 0.05 + d * 0.08))
        
        self.mass = self.dry_mass + self.fuel_mass_initial
        self.fuel_mass = self.fuel_mass_initial
        self.steps = 0

        self.state = np.array([x0, y0, vx0, vy0, theta0, omega0, 1.0], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        self.steps += 1
        x, y, vx, vy, theta, omega, _ = self.state

        # Parse actions
        raw_throttle, raw_gimbal = float(action[0]), float(action[1])
        
        # Throttle mapping: action in [-1, 1] -> [0.0, 1.0]
        # Real rocket engine constraint: If throttle > 0.05, clamp between [min_thrust, 1.0]
        throttle_cmd = (raw_throttle + 1.0) / 2.0
        if throttle_cmd < 0.05 or self.fuel_mass <= 0.0:
            thrust = 0.0
        else:
            effective_throttle = self.min_thrust_ratio + (1.0 - self.min_thrust_ratio) * throttle_cmd
            thrust = effective_throttle * self.max_thrust

        # Gimbal mapping: action in [-1, 1] -> [-max_gimbal, +max_gimbal]
        gimbal_angle = np.clip(raw_gimbal, -1.0, 1.0) * self.max_gimbal

        # Mass depletion (Rocket Equation)
        if thrust > 0.0 and self.fuel_mass > 0.0:
            m_dot = thrust / (self.isp * self.g)
            consumed = m_dot * self.dt
            self.fuel_mass = max(0.0, self.fuel_mass - consumed)
            self.mass = self.dry_mass + self.fuel_mass

        # Moment of Inertia for a slender cylinder rotating about center of mass: I = (1/12) * m * L^2
        I_zz = (1.0 / 12.0) * self.mass * (self.length ** 2)

        # Vector Thrust Forces:
        # Thrust nozzle angle relative to world vertical is (theta + gimbal_angle)
        thrust_world_angle = theta + gimbal_angle
        F_thrust_x = -thrust * np.sin(thrust_world_angle)
        F_thrust_y = thrust * np.cos(thrust_world_angle)

        # Aerodynamic Drag: F_drag = -0.5 * rho * Cd * A * v^2
        rho = 1.225  # kg/m^3 air density at sea level
        Cd = 0.8     # Drag coefficient for rocket cylinder
        Area = np.pi * (self.radius ** 2)
        F_drag_x = -0.5 * rho * Cd * Area * vx * abs(vx)
        F_drag_y = -0.5 * rho * Cd * Area * vy * abs(vy)

        # Net forces and linear accelerations
        ax = (F_thrust_x + F_drag_x) / self.mass
        ay = (F_thrust_y + F_drag_y) / self.mass - self.g

        # Torque: Generated by nozzle gimbal distance to center of mass (L / 2)
        # tau = -thrust * sin(gimbal_angle) * (L / 2)
        torque = -thrust * np.sin(gimbal_angle) * (self.length / 2.0)
        alpha = torque / I_zz

        # Numerical integration (Semi-implicit Euler for dynamic stability)
        vx += ax * self.dt
        vy += ay * self.dt
        x += vx * self.dt
        y += vy * self.dt

        omega += alpha * self.dt
        theta += omega * self.dt
        
        # Keep theta within [-pi, pi]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        fuel_ratio = self.fuel_mass / self.fuel_mass_initial
        self.state = np.array([x, y, vx, vy, theta, omega, fuel_ratio], dtype=np.float32)

        # Determine Termination & Reward
        terminated = False
        truncated = self.steps >= self.max_steps
        
        # Touchdown criteria
        touchdown = y <= 0.0

        # Reward shaping components
        reward = 0.0

        if touchdown:
            terminated = True
            # Check landing tolerances:
            # 1. Vertical speed < 3.0 m/s (ideal < 1.5 m/s)
            # 2. Horizontal speed < 1.5 m/s
            # 3. Tilt angle < 8 degrees
            # 4. Distance to center of pad < 5 meters
            is_upright = abs(theta) < np.radians(8.0)
            is_soft_v = abs(vy) < 3.0
            is_soft_h = abs(vx) < 1.5
            is_on_pad = abs(x) < 6.0

            if is_upright and is_soft_v and is_soft_h and is_on_pad:
                # Spectacular successful landing!
                reward += 500.0
                # Extra bonus for pinpoint accuracy and pristine touchdown
                reward += (5.0 - abs(x)) * 30.0
                reward += (3.0 - abs(vy)) * 30.0
            else:
                # Crash touchdown!
                reward -= 150.0
                if not is_upright:
                    reward -= 50.0  # Tipped over
                if not is_soft_v:
                    reward -= 50.0  # Hit the ground too hard
                if not is_on_pad:
                    reward -= 30.0  # Missed the barge
        elif abs(x) > 250.0 or y > 600.0 or abs(theta) > np.radians(75.0):
            # Rocket flew out of bounds or flipped inverted
            terminated = True
            reward -= 200.0
        else:
            # In-flight step shaping to guide policy convergence
            # 1. Penalize distance from pad
            reward -= 0.01 * (abs(x) / 50.0)
            # 2. Penalize tilt angle (stay upright!)
            reward -= 0.05 * (abs(theta) / np.radians(10.0))
            # 3. Penalize excessive angular velocity (damping)
            reward -= 0.02 * abs(omega)
            # 4. Small penalty for fuel consumption (efficiency incentive)
            reward -= 0.005 * (thrust / self.max_thrust)
            # 5. Guide vertical velocity toward safe landing profile
            if y < 80.0:
                target_vy = -max(1.0, (y / 80.0) * 15.0) # As it gets closer to 0, vy should decelerate to -1 m/s
                reward -= 0.05 * abs(vy - target_vy)

        return self.state, reward, terminated, truncated, {
            "altitude": y,
            "dist_x": x,
            "vy": vy,
            "tilt_deg": np.degrees(theta),
            "fuel_pct": fuel_ratio * 100.0
        }
