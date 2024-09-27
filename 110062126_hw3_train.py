import gym
import torch
import numpy as np
from torch import nn
from gym.spaces import Box
import gym_multi_car_racing
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Normal

import yaml
import wandb
import random
import datetime
from tqdm import trange
from pathlib import Path
from collections import deque

# Wrappers
class RewardModify(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    def reward(self, reward):
        if self.env.driving_on_grass[0] == True:
            return np.array([-0.1], dtype=np.float32)
        return reward

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, end, info = self.env.step(action)
            total_reward += reward
            if end:
                break
        return obs, total_reward, end, info

class CropImage(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        new_shape = list(env.observation_space.shape)
        new_shape[0] -= 12
        self.observation_space = Box(low=0, high=255, shape=tuple(new_shape), dtype=np.uint8)
    def observation(self, obs:np.ndarray):
        obs = obs.copy()[:, :-12, :, :]
        return obs

class GrapScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)
    
    def observation(self, obs:np.ndarray):
        gray = np.dot(obs[..., :], [0.299, 0.587, 0.114])
        gray = gray / 128.0 - 1.0
        return gray

class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, num_stack):
        super(FrameStack, self).__init__(env)
        self.num_stack = num_stack

        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return np.array(list(self.frames))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self.observation()

def CreateEnv():
    env = gym.make("MultiCarRacing-v0", num_agents=1, use_ego_color=True)
    env = RewardModify(env)
    env = SkipFrame(env)                # (num_agents, 96, 96, 3)
    env = CropImage(env)                # (num_agents, 96, 96, 3) -> (num_agents, 84, 96, 3)
    env = GrapScale(env)                # (num_agents, 84, 96, 3) -> (num_agents, 84, 96)
    env = FrameStack(env, num_stack=4)  # (num_agents, 84, 96) -> (4, num_agents, 84, 96)

    # obs.shape         : (4, num_agents, 84, 96)
    # obs_space.shape   : (4, 84, 96)
    # Type              : np.ndarray
    return env

class ReplayBuffer:
    def __init__(self, buffer_size=20000, batch_size=32, random_seed=1, device='cuda'):
        self.buffer         = deque(maxlen=buffer_size)
        self.size           = 0
        self.buffer_size    = buffer_size
        self.batch_size     = batch_size
        self.random_seed    = random.seed(random_seed)
        self.device         = device

    def store(self, state, action, reward, next_state, done):
        experience = self.to_experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.size = min(self.size+1, self.buffer_size)

    def sample_batch(self):
        experiences = random.sample(self.buffer, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e["state"] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e["action"] for e in experiences])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e["reward"] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e["next_state"] for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e["done"] for e in experiences]).astype(np.uint8)).float().to(self.device)
        return states, actions, rewards, next_states, dones

    def to_experience(self, state, action, reward, next_state, done):
        return {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        }

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

    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(self.device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.sum(log_prob, dim=1).unsqueeze(1)
        action = action.cpu() + torch.tensor([0., 1., 1.]) / torch.tensor([1., 2., 2.])
        return action, log_prob

    def get_action(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(self.device)
        action = torch.tanh(mu + e * std)[0].cpu() + torch.tensor([0., 1., 1.]) / torch.tensor([1., 2., 2.])
        action = action.cpu()
        return action

class Critic(nn.Module):
    def __init__(self, state_size, action_size, random_seed, hidden_size=32, init_w=3e-3, device='cuda'):
        super(Critic, self).__init__()
        self.random_seed = torch.manual_seed(random_seed)
        self.cnn = CNNBackbone()
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.init_w = init_w
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.uniform_(self.fc3.weight, a=-self.init_w, b=self.init_w)

    def forward(self, state, action):
        state = self.cnn(state).to(self.device)
        if type(action) == np.ndarray:
            action = torch.from_numpy(action).to(self.device)
        else:
            action = action.to(self.device)
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent():
    def __init__(self, wandb_run, state_size=256, action_size=3, random_seed=1, hidden_size=256, gamma=0.99,
                 lr_act=3e-4, lr_cri=3e-4, buffer_size=20000, batch_size=32, rho=1e-2, device='cuda', max_t=2000):
        self.wandb_run = wandb_run
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed
        self.seed = random.seed(random_seed)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma

        # For alpha in policy udpate, we apply non-fixed appraoch
        # Therefore we nned targte entropy here
        # https://docs.cleanrl.dev/rl-algorithms/sac/#explanation-of-the-logged-metrics
        self.target_entropy = -action_size
        self.alpha = torch.tensor(1).to(self.device)
        self.rho = rho
        self.log_alpha = torch.tensor([lr_act], requires_grad=True)
        self.alpha_optimizer = Adam(params=[self.log_alpha], lr=lr_act)

        # Actor Network 
        self.actor = Actor(state_size, action_size, random_seed, hidden_size, device=device).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_act)     
        
        # Critic Network
        self.critic1 = Critic(state_size, action_size, random_seed, hidden_size, device=device).to(device)
        self.critic2 = Critic(state_size, action_size, random_seed, hidden_size, device=device).to(device)
        # Critic Targets
        self.critic1_target = Critic(state_size, action_size, random_seed,hidden_size, device=device).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        
        self.critic2_target = Critic(state_size, action_size, random_seed,hidden_size, device=device).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=lr_cri, weight_decay=0)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=lr_cri, weight_decay=0) 

        # Replay memory
        self.buffer = ReplayBuffer(buffer_size, batch_size, random_seed, device)

    def step(self, state, action, reward, next_state, done, timestep, episode):
        self.buffer.store(state, action, reward, next_state, done)
        if self.buffer.size > self.batch_size:
            experiences = self.buffer.sample_batch()
            self.learn(timestep, experiences, episode)
    
    def act(self, state:np.ndarray):
        action = self.actor.get_action(state).detach()
        return action
    
    def learn(self, timestep, experiences, episode):
        states, actions, rewards, next_states, dones = experiences

        # Step 12: Compute targets for Q functions (critics)
        # y(r, s', d) = r + γ(1-d)(min_{i=1,2}{Q_target_{i}(s', \tilde{a}')} - α\log{π_θ(\tilde{a}' | s')})
        # Where \tilde{a}' ~ π_θ(·|s')
        next_action, nxta_log_prob = self.actor.evaluate(next_states)
        Q_target1 = self.critic1_target(next_states, next_action)
        Q_target2 = self.critic2_target(next_states, next_action)
        Q_target_min = torch.min(Q_target1, Q_target2).to(self.device)
        nxta_log_prob = nxta_log_prob.to(self.device)
        y = rewards.to(self.device) + self.gamma * (1 - dones.to(self.device)) * (Q_target_min - self.alpha * nxta_log_prob)
        
        # Step 13: Update Q functions (Critics) by one step of gradient descent using
        # ∇_Φ_i 1/|B| * \sum_{(s,a,r,s',d) \in B}{ (Q_Φ_i(s, a) - y(r, s', d))^2 }
        # a.k.a MSE Loss
        Q1 = self.critic1(states, actions)
        Q2 = self.critic2(states, actions)
        
        critic_loss1 = F.mse_loss(Q1, y.detach())
        critic_loss2 = F.mse_loss(Q2, y.detach())
        
        self.critic1_optimizer.zero_grad()
        critic_loss1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()

        # Step 14: Update policy (Actor) by one step of gradient "ascent" using
        # ∇_Φ_i 1/|B| * \sum_{s \in B}{ (\min_{i=1,2}{Q_Φ_i(s, \tilde{a}_θ(s))} - α\log{π_θ(\tilde{a}_θ(s) | s)}) }
        # Where \tilde{a}_θ(s) is sample from π_θ(·|s)

        # But since we're using non-fixed alpha, so update it first
        # J(α) = E_{a_t \sim π_t}[-α\log{π_t(a_t|s_t)} - α\bar{H}]
        alpha = torch.exp(self.log_alpha).to(self.device)
        actions_pred, pred_action_log_prob = self.actor.evaluate(states)
        alpha_loss = - (self.log_alpha.cpu() * (pred_action_log_prob.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = alpha
        # Add a negative on loss, since we want gradient ascent
        Q1 = self.critic1(states, actions_pred.squeeze(0))
        Q2 = self.critic2(states, actions_pred.squeeze(0))
        Q_min = torch.min(Q1, Q2)
        actor_loss = -(Q_min - alpha * pred_action_log_prob.squeeze(0)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Step 15: Update target network woth
        # Φ_{target_i} ← ρΦ_{target_i} + (1 - ρ)Φ_i
        for target_param, local_param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.rho*local_param.data + (1.0-self.rho)*target_param.data)
        for target_param, local_param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.rho*local_param.data + (1.0-self.rho)*target_param.data)
        
        # log to wandb
        self.wandb_run.log({
            'timestep': timestep,
            'episode': episode,
            'actor_loss': actor_loss.detach().float().cpu().mean(),
            'critic1_loss': critic_loss1.detach().float().cpu().mean(),
            'critic2_loss': critic_loss2.detach().float().cpu().mean(),
            # 'rewards': torch.mean(rewards.detach()).item()
        })

def save_ckpt(save_dir, episode, agent:Agent):
    # Save actor module, actor optimizer, critic module, critic optimizer
    torch.save({
        'episode': episode,
        'actor_model': agent.actor.state_dict(),
        'actor_optimizer': agent.actor_optimizer.state_dict(),
        'critic1_model': agent.critic1.state_dict(),
        'critic1_optimizer': agent.critic1_optimizer.state_dict(),
        'critic2_model': agent.critic2.state_dict(),
        'critic2_optimizer': agent.critic2_optimizer.state_dict(),
        'critic1_target_model': agent.critic1_target.state_dict(),
        'critic2_target_model': agent.critic2_target.state_dict(),
        'alpha': agent.alpha,
        'log_alpha': agent.log_alpha,
        'alpha_optimizer': agent.alpha_optimizer.state_dict()
    }, save_dir / f'{episode}.pth')

def load_ckpt(save_dir, episode, agent:Agent):
    data = torch.load(open(save_dir / f'{episode}.pth', 'rb'))
    agent.actor.load_state_dict(data['actor_model'])
    agent.actor_optimizer.load_state_dict(data['actor_optimizer'])
    agent.critic1.load_state_dict(data['critic1_model'])
    agent.critic1_optimizer.load_state_dict(data['critic1_optimizer'])
    agent.critic2.load_state_dict(data['critic2_model'])
    agent.critic2_optimizer.load_state_dict(data['critic2_optimizer'])
    agent.critic1_target.load_state_dict(data['critic1_target_model'])
    agent.critic2_target.load_state_dict(data['critic2_target_model'])
    agent.alpha = data['alpha']
    agent.log_alpha = data['log_alpha']
    agent.alpha_optimizer.load_state_dict(data['alpha_optimizer'])

def play(env, agent:Agent):
    agent.actor.eval()
    rewards = 0

    state = env.reset()
    state = np.expand_dims(state.squeeze(), axis=0)

    while True:
        action = agent.act(torch.from_numpy(state).float()).numpy()
        next_state, reward, done, info = env.step(action)
        next_state = np.expand_dims(next_state.squeeze(), axis=0)
        state = next_state
        rewards += reward.item()
        env.render()
        
        if done:
            break
    agent.actor.train()
    return rewards

if __name__ == '__main__':
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    # Finetune
    # save_dir = Path("checkpoints") / "2024-04-20T02-32-41"
    episode = 1

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    scores_deque = deque(maxlen=100)
    n_episodes = 10000
    print_every = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run = wandb.init(
        project="DRL-HW3-SAC",
        # Track hyperparameters and run metadata
        config={
            "random_seed"       : config["random_seed"],
            "hidden_size"       : config["hidden_size"],
            "lr_act"            : config["gamma"],
            "lr_cri"            : config["lr_cri"],
            "buffer_size"       : config["buffer_size"],
            "batch_size rate"   : config["batch_size"],
            "rho"               : config["rho"],
            "gamma"             : config["gamma"],
            "max_t"             : config["max_t"]
        },
    )

    env = CreateEnv()
    env.reset()
    env.close()
    agent = Agent(**config, device=device, wandb_run=run)
    # load_ckpt(save_dir, episode, agent)

    # Training Loop
    for i_episode in trange(episode, n_episodes+1):
        state = env.reset()
        state = np.expand_dims(state.squeeze(), axis=0)
        score = 0
        for t in range(config['max_t']):
            action = agent.act(torch.from_numpy(state).float()).numpy()
            next_state, reward, done, info = env.step(action)
            next_state = np.expand_dims(next_state.squeeze(), axis=0)
            agent.step(state, action, reward, next_state, done, t, i_episode)
            state = next_state
            score += reward.item()

            if done:
                break 
        
        scores_deque.append(score)
        run.log({
            'episode': i_episode,
            'reward': score,
            'avg_reward': np.mean(scores_deque)
        })
        
        print('\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)))
        if i_episode % print_every == 0:
            save_ckpt(save_dir, i_episode, agent)
            eval_score = play(env=env, agent=agent)
            print('Episode {} Eval score: {:.2f}'.format(i_episode, eval_score))
            run.log({
                'episode': i_episode,
                'eval_reward': eval_score
            })
