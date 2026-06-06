import time

import torch
import torchvision.transforms as transforms
import numpy as np

import environment


class DRQNEnv(environment.BaseEnv):
    def __init__(self, n_actions=8, **kwargs) -> None:
        super().__init__(**kwargs)
        # divide the action space into n_actions
        self._n_actions = n_actions
        self._delta = 0.05

        theta = np.linspace(0, 2*np.pi, n_actions, endpoint=False) # if endpoint=True, the last action would be the same as the first one (0 and 2*pi)
        actions = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        self._actions = {i: action for i, action in enumerate(actions)}

        self._goal_thresh = 0.05
        self._max_timesteps = 50

        # --- Reward shaping parameters (potential-based progress shaping) ---
        # We reward the *reduction* in distance each step, not the absolute
        # distance. This is something the agent can actually control, it is
        # telescoping (total reward depends only on net progress, so it is
        # scale-stable), and it keeps per-step rewards O(1) instead of O(-100).
        self._success_reward = 10.0
        self._w_reach = 5.0    # weight on end-effector -> object progress
        self._w_push = 20.0    # weight on object -> goal progress (the real task)
        self._time_penalty = 0.01
        # Distances at the start of the current step, filled in by step().
        self._prev_ee_to_obj = 0.0
        self._prev_obj_to_goal = 0.0
        # Minimum object->goal distance at spawn so every episode needs a real push.
        self._min_spawn_dist = 0.15
        # Curriculum: the MAX object->goal spawn distance. Training ramps this up
        # from easy (short pushes) to hard. Defaults to full table difficulty so
        # testing / standalone use is unaffected.
        self._curriculum_max_dist = 0.6

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        # getattr fallbacks: _create_scene runs once from super().__init__()
        # before DRQNEnv.__init__ has set these attributes.
        min_dist = getattr(self, "_min_spawn_dist", 0.15)
        max_dist = getattr(self, "_curriculum_max_dist", 0.6)
        max_dist = max(max_dist, min_dist + 0.01)
        # Place the goal anywhere on the table, then place the object at a
        # controlled distance/angle from it. The required push distance is thus
        # always in [min_dist, max_dist] (no free successes), and the curriculum
        # can make early episodes easy by keeping max_dist small.
        x_lo, x_hi, y_lo, y_hi = 0.25, 0.75, -0.3, 0.3
        while True:
            goal_pos = [np.random.uniform(x_lo, x_hi),
                        np.random.uniform(y_lo, y_hi),
                        1.025]
            dist = np.random.uniform(min_dist, max_dist)
            ang = np.random.uniform(0, 2 * np.pi)
            ox = goal_pos[0] + dist * np.cos(ang)
            oy = goal_pos[1] + dist * np.sin(ang)
            if x_lo <= ox <= x_hi and y_lo <= oy <= y_hi:
                obj_pos = [ox, oy, 1.5]
                break
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def _get_raw_state(self):
        """Internal helper to get raw physical meters for reward calculations"""
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def high_level_state(self):
        """The 'Sensor' for the neural network, strictly normalized to [-1, 1]"""
        raw_state = self._get_raw_state()
        
        # Min-Max Normalization based on Table Bounds
        # x_bounds = [0.2, 1.2], y_bounds = [-0.5, 0.5]
        min_bounds = np.array([0.2, -0.5, 0.2, -0.5, 0.2, -0.5])
        max_bounds = np.array([1.2,  0.5, 1.2,  0.5, 1.2,  0.5])
        
        normalized_state = (raw_state - min_bounds) / (max_bounds - min_bounds)

        # Safety clip to protect against physics engine clipping or collisions
        normalized_state = np.clip(normalized_state, 0.0, 1.0)
        
        # Scale to [-1, 1] for better neural network dynamics
        scaled_state = (normalized_state * 2.0) - 1.0
        
        return scaled_state
    
    def raw_object_goal_distance(self):
        """Object->goal distance in raw meters (for logging / model selection).
        high_level_state() is normalized to [-1, 1], so distances computed from
        it are NOT in meters and are not comparable to self._goal_thresh."""
        raw = self._get_raw_state()
        return float(np.linalg.norm(raw[2:4] - raw[4:6]))
    
    def is_terminal(self):
        return self.raw_object_goal_distance() < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def reward(self):
        # Calculate rewards using RAW physics meters, not neural net inputs.
        # NOTE: step() must set self._prev_ee_to_obj / self._prev_obj_to_goal
        # to the distances measured BEFORE this step's motion.
        state = self._get_raw_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]
        
        # Standard Euclidean distances (meters), measured AFTER the motion.
        ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        obj_to_goal = np.linalg.norm(obj_pos - goal_pos)

        # Success: big terminal bonus.
        if obj_to_goal < self._goal_thresh:
            return self._success_reward

        # Potential-based progress shaping: reward the *reduction* in distance
        # this step. Positive when the agent makes progress, negative when it
        # backslides. This is the part of the signal the agent can actually
        # control, and (being telescoping) the episode return depends only on
        # net progress, keeping Q-values small and stable.
        r_reach = self._w_reach * (self._prev_ee_to_obj - ee_to_obj)
        r_push = self._w_push * (self._prev_obj_to_goal - obj_to_goal)
        r_time = -self._time_penalty

        return r_reach + r_push + r_time

    def set_curriculum_max_dist(self, d):
        """Set the maximum object->goal spawn distance (curriculum difficulty).
        Call BEFORE reset(), since the spawn happens during reset()."""
        self._curriculum_max_dist = float(np.clip(d, self._min_spawn_dist + 0.01, 0.8))

    def step(self, action_id):
        # Capture distances BEFORE moving, so reward() can measure progress.
        prev = self._get_raw_state()
        self._prev_ee_to_obj = np.linalg.norm(prev[:2] - prev[2:4])
        self._prev_obj_to_goal = np.linalg.norm(prev[2:4] - prev[4:6])

        action = self._actions[action_id] * self._delta
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.06]])
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
        self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
        self._t += 1

        state = self.high_level_state()
        reward = self.reward()
        terminal = self.is_terminal()
        truncated = self.is_truncated()
        return state, reward, terminal, truncated


if __name__ == "__main__":
    N_ACTIONS = 8
    env = DRQNEnv(n_actions=N_ACTIONS, render_mode="gui")
    for episode in range(10):
        env.reset()
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        start = time.time()
        while not done:
            action = np.random.randint(N_ACTIONS)
            state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            cumulative_reward += reward
            episode_steps += 1
        end = time.time()
        print(f"Episode={episode}, reward={cumulative_reward}, RPS={cumulative_reward/episode_steps}")
