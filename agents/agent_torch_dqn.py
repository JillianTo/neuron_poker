"""Player based on a trained neural network"""
import math
import random
from collections import namedtuple, deque
from itertools import count

from gym_env.enums import Action

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(n_observations, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, n_actions),
                    )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.model(x)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='DQN'):
        """Initiaization of an agent"""
        self.name = name

    def initiate_agent(self, env, lr=1e-4, path=None):
        self.env = env
        # if GPU is to be used
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        state, info = self.env.reset()
        self.n_observations = env.observation_space[0]

        self.policy_net = DQN(self.n_observations, n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

        if path != None:
            self.policy_net.load_state_dict(torch.load(path+f'/policy_net.pth', weights_only=True))
            self.target_net.load_state_dict(torch.load(path+f'/target_net.pth', weights_only=True))
            self.optimizer.load_state_dict(torch.load(path+f'/optimizer.pth', weights_only=True))

        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def process_action(self, action, info):
        """Find nearest legal action"""
        if 'legal_moves' in info.keys():
            legal_moves_limit = info['legal_moves']
        else:
            legal_moves_limit = None
    
        if legal_moves_limit is not None:
            legal_moves_limit = [move.value for move in legal_moves_limit]
            if action not in legal_moves_limit:
                curr_action = action
                for i in range(7):
                    curr_action = action+i
                    if curr_action in legal_moves_limit:
                        action = curr_action
                        break
                    curr_action = action-i
                    if curr_action in legal_moves_limit:
                        action = curr_action
                        break
                
        return action

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.policy_net(observation).max(1).indices.view(1, 1)
        else:
            action = torch.tensor([[action_space.sample()]], device=self.device, dtype=torch.long)

        return self.process_action(action, info).to(self.device)

    def optimize_model(self):
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        BATCH_SIZE = 128
        GAMMA = 0.99

        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def save_states(self, path='.'):
        torch.save(self.policy_net.state_dict(), path+f'/policy_net.pth')
        torch.save(self.target_net.state_dict(), path+f'/target_net.pth')
        torch.save(self.optimizer.state_dict(), path+f'/optimizer.pth')

    def train(self, num_episodes=50):
        # TAU is the update rate of the target network
        TAU = 0.005

        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = torch.zeros(1, self.n_observations)
            for t in count():
                action = self.action(self.env.action_space, state.to(self.device), info)
                observation, reward, done, info = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state.to(self.device), action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    break

        self.save_states()
        print('Complete')

    def play(self, num_episodes=50):
        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = torch.zeros(1, self.n_observations)
            for t in count():
                action = self.action(self.env.action_space, state.to(self.device), info)
                observation, reward, done, info = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                if done:
                    break
                else:
                    state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        print('Complete')
