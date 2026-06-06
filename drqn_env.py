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

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
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
        raw = self._get_raw_state()
        return float(np.linalg.norm(raw[2:4] - raw[4:6]))
    
    def is_terminal(self):
        return self.raw_object_goal_distance() < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def reward(self):
        # Calculate rewards using RAW physics meters, not neural net inputs
        state = self._get_raw_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]
        
        # Calculate standard Euclidean distances (meters)
        ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        obj_to_goal = np.linalg.norm(obj_pos - goal_pos)
        
        # Check success condition
        success = obj_to_goal < self._goal_thresh
        
        if success:
            # Balanced completion bonus. Strong enough to pull the agent in, but not massive enough to break the DDQN Q-value updates.
            return 10.0
            
        # Dense Continuous Guidance (Negative distances)
        # Scaled to smoothly guide the end-effector to the object, and object to goal.
        r_reach = -1.0 * ee_to_obj
        r_push = -2.0 * obj_to_goal
        
        # Effective Time Penalty (Step Tax)
        # Changed from -0.05 to -0.5. Over 50 steps, this adds up to -25.0.
        # This scale forces the network to actively care about minimizing steps.
        r_time = -0.5
        
        return r_reach + r_push + r_time

    def step(self, action_id):
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
