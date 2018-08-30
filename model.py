import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
	"""
	"""
	def __init__(self, n_state, n_action, seed, units_fc1=128, units_fc2=128):
		"""

		"""
		super(QNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(n_state, units_fc1)
		self.fc2 = nn.Linear(units_fc1, units_fc2)
		self.fc3 = nn.Linear(units_fc2, n_action)


	def forward(self, state):
		"""
		"""
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		return self.fc3(x)