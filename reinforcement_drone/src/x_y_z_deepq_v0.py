#!/usr/bin/env python

import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from task_env.x_y_z_line_follow import XYZLineFollowEnv
import gym
import numpy as np
from collections import deque
import random
import rospy
import rospkg
from data_gatherer import DataGatherer
import time
import datetime


rospy.init_node('x_y_z_deepq', anonymous=True, log_level=rospy.ERROR)

tf.keras.backend.set_floatx('float64')
wandb.init(name='DQN', project="deep-rl-xyz")

gamma = rospy.get_param("/drone/gamma")
lr = float(rospy.get_param("/drone/lr"))
batch_size = rospy.get_param("/drone/batch_size")
eps = rospy.get_param("/drone/eps")
eps_decay = rospy.get_param("/drone/eps_decay")
eps_min = rospy.get_param("/drone/eps_min")
nepisodes = rospy.get_param("/drone/nepisodes")
check_rate = rospy.get_param("/drone/check_rate")


rospack = rospkg.RosPack()
rospack.list()

pkg_path = rospack.get_path('reinforcement_drone')
outdir = pkg_path + '/training_results/deepq/x_y_z'
dataGatherer = DataGatherer(outdir)
dataGatherer.create_checkpoint()

start_time = time.time()


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)

class ActionStateModel:
    def __init__(self, state_dim, action_dim):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.epsilon = eps
        self.model = self.create_model()
    
    def create_model(self):
        model = tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim)
        ])
        model.compile(loss='mse', optimizer=Adam(lr))
        return model
    
    def predict(self, state):
        return self.model.predict(state)
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= eps_decay
        self.epsilon = max(self.epsilon, eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)
    

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)
    
    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(batch_size), actions] = rewards + (1-done) * next_q_values * gamma
            self.model.train(states, targets)
    
    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            steps_count = 0
            info = {}

            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.buffer.put(state, action, reward*0.01, next_state, done)
                total_reward += reward
                state = next_state
                steps_count = steps_count+1
            if self.buffer.size() >= batch_size:
                self.replay()
            self.target_update()
            print('EP{} EpisodeReward={}'.format(ep, total_reward))
            wandb.log({'Reward': total_reward})

            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)
            dataGatherer.add_to_checkpoint(
                ep+1 + dataGatherer.start_episode_number,
                steps_count,
                total_reward,
                0,
                0,
                round(self.model.epsilon, 2),
                datetime.timedelta(hours=h, minutes=m, seconds=s),
                info
            )

            if (ep % check_rate == 0):
                dataGatherer.create_checkpoint()


def main():
    env = gym.make('XYZLineFollowEnv-v0')
    agent = Agent(env)

    agent.train(max_episodes=5000)

if __name__ == "__main__":
    main()
    