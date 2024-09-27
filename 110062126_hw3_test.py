import gym
import torch
import numpy as np
from torch import nn
from gym.spaces import Box
import gym_multi_car_racing
import torch.nn.functional as F
from torch.distributions import Normal

import random
from tqdm import trange
from pathlib import Path
from collections import deque

def CreateEnv():
    env = gym.make("MultiCarRacing-v0", num_agents=1, use_ego_color=True)
    return env

class CNNBackbone(nn.Module):
    def __init__(self, input_dim=(4, 84, 96), output_dim=256):
        super(CNNBackbone, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = self.build_net()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, input):
        # output: torch.Size([batch_size, 256])
        if input.dim() == 3:
            input = input.unsqueeze(0)
        assert(input.dim() == 4)
        input = input.to(self.device)
        return self.net(input)

    def build_net(self):
        c = self.input_dim[0]
        cnn = torch.nn.Sequential()

        # Convolution 1
        conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=4, stride=2)
        nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        cnn.add_module("conv_1", conv1)
        cnn.add_module("relu_1", nn.LeakyReLU())

        # Convolution 2
        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2)
        nn.init.kaiming_normal_(conv2.weight, mode='fan_out', nonlinearity='leaky_relu')
        cnn.add_module("conv_2", conv2)
        cnn.add_module("maxpool1", nn.MaxPool2d(kernel_size=2))

        # Convolution 3
        conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1)
        nn.init.kaiming_normal_(conv3.weight, mode='fan_out', nonlinearity='leaky_relu')
        cnn.add_module("conv_3", conv3)
        cnn.add_module("maxpool2", nn.MaxPool2d(kernel_size=2))

        # Reshape CNN output
        cnn.add_module("flatten", torch.nn.Flatten())

        # # Calculate input size
        state = torch.zeros(*(self.input_dim))
        dims = cnn(state)
        line_input_size = int(np.prod(dims.size()))

        # Linear 1
        line1 = nn.Linear(line_input_size, 256)
        nn.init.kaiming_normal_(line1.weight, mode='fan_out', nonlinearity='relu')
        cnn.add_module("line_1", line1)
        cnn.add_module("relu_2", nn.LeakyReLU())

        return cnn
    
class Actor(nn.Module):
    def __init__(self, state_size, action_size, random_seed, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2, device='cuda'):
        super(Actor, self).__init__()
        self.random_seed = torch.manual_seed(random_seed)
        self.init_w = init_w
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device

        self.cnn = CNNBackbone()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.uniform_(self.mu.weight, a=-self.init_w, b=self.init_w)
        nn.init.uniform_(self.log_std_linear.weight, a=-self.init_w, b=self.init_w)
    
    def forward(self, state):
        x = self.cnn(state).to(self.device)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def get_action(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(self.device)
        action = torch.tanh(mu + e * std)[0].cpu() + torch.tensor([0., 1., 1.]) / torch.tensor([1., 2., 2.])
        action = action.cpu()
        return action

class Agent():
    def __init__(self):
        self.state_size = 256
        self.action_size = 3
        self.random_seed = 1
        self.hidden_size = 256
        self.seed = random.seed(self.random_seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.frame_skip = 0
        self.frames = deque(maxlen=4)
        self.last_action = None
        # Actor Network 
        self.actor = Actor(self.state_size, self.action_size, self.random_seed, self.hidden_size, device=self.device).to(self.device)
        self.load()

    def load(self, filepath='./110062126_hw3_data'):
        data = torch.load(open(filepath, 'rb'))
        self.actor.load_state_dict(data)
    
    def act(self, observation):
        if self.frame_skip % 4 == 0:
            # Crop Image
            state = observation.copy()[:, :-12, :, :]
            # Gray Scale
            state = np.dot(state[..., :], [0.299, 0.587, 0.114])
            state = state / 128.0 - 1.0
            while len(self.frames) < 4:
                self.frames.append(state)
            self.frames.append(state)
            state = np.array(list(self.frames))
            state = np.expand_dims(state.squeeze(), axis=0)
            state = torch.from_numpy(state).float()
            action = self.actor.get_action(state).detach().numpy()
            self.last_action = action
        self.frame_skip += 1
        return self.last_action


if __name__ == '__main__':
    env = CreateEnv()
    env.reset()
    env.close()
    agent = Agent()
    total_score = 0

    for i_episode in trange(1, 50+1):
        state = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            score += reward.item()
            # env.render()
        total_score += score
        print('\rEpisode {} Reward: {:.2f}'.format(i_episode, score))
    print('Average score: {}'.format(total_score / 50.0))
