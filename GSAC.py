from model import PolicyNetContinuous, Double_Q_Critic
import torch.nn.functional as F
import numpy as np
import torch
import copy


class GSAC_agent():
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
		self.policy_noise = 0.2*self.action_bound
		self.noise_clip = 0.5*self.action_bound
		self.tau = 0.005
		self.delay_counter = 0
		self.gmm_k = np.clip(int(np.log2(self.action_dim))-1, a_min=1, a_max=4)
		print('number of k:', self.gmm_k)
        
		self.actor = PolicyNetContinuous(self.state_dim, self.hidden_dim, self.action_dim, self.action_bound, self.gmm_k).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
		
		self.log_alpha = torch.tensor(np.log(0.001), dtype=torch.float)
		self.log_alpha.requires_grad = True 
		self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
		self.target_entropy = 0.2*np.log(self.action_dim)
		
		self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.critic_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.device)
		
	def train(self):
		self.delay_counter += 1
		with torch.no_grad():
			s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
			# Compute the target Q
			target_a_noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			next_a, entropy = self.actor(s_next)
			entropy = entropy.unsqueeze(1)
			smoothed_target_a = (next_a + target_a_noise).clamp(-self.action_bound, self.action_bound)
			target_Q1, target_Q2 = self.q_critic_target(s_next, smoothed_target_a)
			soft_Q = torch.min(target_Q1, target_Q2) + self.log_alpha.exp() * entropy
			target_Q = r + (~dw) * self.gamma * soft_Q  #dw: die or win

		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)
		#print(entropy.shape, target_Q.shape, current_Q1.shape, current_Q2.shape)
		#raise ValueError("Division by zero is not allowed.")
		# Compute critic loss, and Optimize the q_critic
		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		q_loss.backward()
		self.q_critic_optimizer.step()
		
        # alpha loss
		'''
		alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
		self.log_alpha_optimizer.zero_grad()
		alpha_loss.backward()
		
		if self.log_alpha.grad is not None:
			print("Gradient:", self.log_alpha.grad.item())
		else:
			print("No gradient for log_alpha")
		
		self.log_alpha_optimizer.step()
		'''
        
		if self.delay_counter > self.delay_freq:
			# Update Actor
			new_actions, new_entropy = self.actor(s)
			new_Q1, new_Q2 = self.q_critic_target(s, new_actions)
			actor_loss = torch.mean(self.log_alpha.exp() * (-new_entropy) - torch.min(new_Q1, new_Q2))
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			'''
			for name, param in self.actor.named_parameters():
				if param.grad is not None:
					print(f"Gradient for {name}: {param.grad.norm().item()}") 
				else:
					print(f"No gradient for {name}")
			raise ValueError('sth wrong here')
			'''
			self.actor_optimizer.step()

			# Update the frozen target models
			with torch.no_grad():
				for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			self.delay_counter = 0

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/GSAC/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.q_critic.state_dict(), "./model/GSAC/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load("./model/GSAC/{}_actor{}.pth".format(EnvName, timestep)))
		self.q_critic.load_state_dict(torch.load("./model/GSAC/{}_q_critic{}.pth".format(EnvName, timestep)))


class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size, dvc):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = torch.from_numpy(a).to(self.dvc) # Note that a is numpy.array
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]



