import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
	""" Deep Q-learning agent class.
	"""
	def __init__(self, dim_state, n_action, seed):
		"""Initialize an Agent object.

		Params
		------



		"""
		self.dim_state = dim_state
		self.n_action = n_action
		self.seed = random.seed(seed)

		# Q-network
		self.qnetwork_online = QNetwork(dim_state, n_action, seed).to(device)
		self.qnetwork_target = QNetwork(dim_state, n_action, seed).to(device)
		self.optimizer = optim.Adam(self.qnetwork_online.parameters(), lr=LR)

		# Replay memory
		self.memory = ReplayBuffer(n_action, BUFFER_SIZE, BATCH_SIZE, seed)
		self.t_step = 0

	def step(self, state, action, reward, next_state, done):
		"""
		"""
		# Save experience in replay memory
		self.memory.add(state, action, reward, next_state, done)

		# Learn every UPDATE_EVERY time steps
		self.t_step = (self.t_step + 1) % UPDATE_EVERY
		if self.t_step == 0:
			if len(self.memory) > BATCH_SIZE:
				experiences = self.memory.sample()
				self.learn(experiences, GAMMA)

	def act(self, state, eps=0.):
		"""
		"""
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.qnetwork_online.eval()
		with torch.no_grad():
			action_values = self.qnetwork_online(state)
		self.qnetwork_online.train()

		# Epsilon greedy policy
		if random.random() > eps:
			return np.argmax(action_values.cpu().data.numpy())
		else:
			return random.choice(np.arange(self.n_action))

	def learn(self, experiences, gamma):
		"""
		"""
		states, actions, rewards, next_states, dones = experiences

		# calc q-target
		q_target_next = self.qnetwork_target(next_states).detach().max(1)[0]\
			.unsqueeze(1)
		q_target = rewards + (gamma * q_target_next * (1 - dones))

		# calc q-expected
		q_expected = self.qnetwork_online(states).gather(1, actions)

		# compute loss
		loss = F.mse_loss(q_expected, q_target)

		# optimize
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# update target network
		self.soft_update(self.qnetwork_online, self.qnetwork_target, TAU)

	def soft_update(self, online_model, target_model, tau):
		"""
		"""
		for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
			target_param.data.copy_(tau*online_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer():
	"""
	"""
	def __init__(self, n_action, buffer_size, batch_size, seed):
		"""
		"""
		self.n_action = n_action
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple("Experience",
			field_names=["state", "action", "reward", "next_state", "done"])
		self.seed = random.seed(seed)

	def add(self, state, action, reward, next_state, done):
		"""
		"""
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self):
		"""
		"""
		experiences = random.sample(self.memory, k=self.batch_size)
		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

		return (states, actions, rewards, next_states, dones)

	def __len__(self):
		"""Return the current size of internal memory.
		"""
		return len(self.memory)