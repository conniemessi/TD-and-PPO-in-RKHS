import gym
    
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F

import numpy as np
import datetime
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class SharedBase(nn.Module):
    """Common feature extractor shared by actor and critic."""
    def __init__(self, state_dim, hidden_dim=64, activation_fn=nn.ReLU):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
        ).float().to(device)

    def forward(self, x):
        return self.layers(x)


class Actor_Model(nn.Module):
    """Actor shares logits layer with critic; only applies Softmax."""
    def __init__(self, shared_base: SharedBase, logits_layer: nn.Linear):
        super().__init__()
        self.base = shared_base  # shared parameters
        self.logits = logits_layer  # shared linear layer

    def forward(self, states):
        features = self.base(states)
        logits = self.logits(features)
        return torch.softmax(logits, dim=-1)


class Critic_Model(nn.Module):
    """Critic shares the same logits layer; outputs raw Q-values/logits."""
    def __init__(self, shared_base: SharedBase, logits_layer: nn.Linear):
        super().__init__()
        self.base = shared_base
        self.logits = logits_layer

    def forward(self, states):
        features = self.base(states)
        return self.logits(features)  # (batch, action_dim)

class Memory(Dataset):
    def __init__(self):
        self.actions        = [] 
        self.states         = []
        self.rewards        = []
        self.dones          = []     
        self.next_states    = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32), np.array(self.actions[idx], dtype = np.float32), np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), np.array(self.next_states[idx], dtype = np.float32)      

    def save_eps(self, state, action, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)        

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]  

class Discrete():
    def sample(self, datas):
        distribution = Categorical(datas)
        return distribution.sample().float().to(device)
        
    def entropy(self, datas):
        distribution = Categorical(datas)    
        return distribution.entropy().float().to(device)
      
    def logprob(self, datas, value_data):
        distribution = Categorical(datas)
        return distribution.log_prob(value_data).unsqueeze(1).float().to(device)

    def kl_divergence(self, datas1, datas2):
        distribution1 = Categorical(datas1)
        distribution2 = Categorical(datas2)

        return kl_divergence(distribution1, distribution2).unsqueeze(1).float().to(device)  

class PolicyFunction():
    def __init__(self, gamma = 0.99, lam = 0.95):
        self.gamma  = gamma
        self.lam    = lam

    def monte_carlo_discounted(self, rewards, dones):
        running_add = 0
        returns     = []        
        
        for step in reversed(range(len(rewards))):
            running_add = rewards[step] + (1.0 - dones[step]) * self.gamma * running_add
            returns.insert(0, running_add)
            
        return torch.stack(returns)
      
    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1 - done) * self.gamma * next_value           
        return q_values

class RKHS_NPG():
    def __init__(self, policy_params, vf_loss_coef, entropy_coef,
                 gamma, lam):
        """Policy-gradient loss with KL-divergence penalty."""
        self.policy_params      = policy_params
        self.vf_loss_coef       = vf_loss_coef
        self.entropy_coef       = entropy_coef

        self.distributions      = Discrete()
        self.policy_function    = PolicyFunction(gamma, lam)


    def compute_loss(self, action_probs, old_action_probs, q_values, old_q_values, next_q_values,
                    actions, rewards, dones):
        """Loss without explicit advantage – uses π_new·Q surrogate and TD critic loss."""
        gamma = self.policy_function.gamma

        # ---------- Policy surrogate ----------
        # KL divergence for NPG: KL(π_new || π_old)
        Kl = self.distributions.kl_divergence(action_probs, old_action_probs)  # KL(new||old)

        # Expected Q under current policy
        expected_q_new = (action_probs * q_values.detach()).sum(dim=1, keepdim=True)  # (B,1)

        pg_targets = expected_q_new - (1.0 / self.policy_params) * Kl  # (B,1)
        pg_loss    = -pg_targets.mean()  # maximise via gradient ascent

        # ---------- Critic TD loss ----------
        # Compute target Q for taken actions
        with torch.no_grad():
            v_next = (action_probs * next_q_values.detach()).sum(dim=1)  # (B,)
            target_q = rewards.squeeze(1) + (1 - dones.squeeze(1)) * gamma * v_next  # (B,)

        actions_long = actions.long()
        q_taken = q_values.gather(1, actions_long.unsqueeze(1)).squeeze(1)  # (B,)

        critic_loss = F.mse_loss(q_taken, target_q)

        # Entropy regularisation
        dist_entropy = self.distributions.entropy(action_probs).mean()

        loss = pg_loss + self.vf_loss_coef * critic_loss - self.entropy_coef * dist_entropy
        return loss, critic_loss.item()

class Agent():  
    def __init__(self, state_dim, action_dim, is_training_mode, policy_params, entropy_coef, vf_loss_coef,
                 batchsize, T, gamma, lam, learning_rate, schedule_pow):
        self.policy_params      = policy_params  # β will be overwritten by schedule
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.batchsize          = batchsize       
        self.T                  = T
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim               
        self.schedule_pow       = schedule_pow

        # Shared feature extractor
        shared_base_current = SharedBase(state_dim)
        shared_base_old     = SharedBase(state_dim)

        logits_layer_current = nn.Linear(64, action_dim).float().to(device)
        logits_layer_old     = nn.Linear(64, action_dim).float().to(device)

        self.actor          = Actor_Model(shared_base_current, logits_layer_current)
        self.actor_old      = Actor_Model(shared_base_old, logits_layer_old)

        self.critic         = Critic_Model(shared_base_current, logits_layer_current)
        self.critic_old     = Critic_Model(shared_base_old, logits_layer_old)

        # Single optimizer for shared parameters (base + logits)
        self.optimizer      = Adam(list(shared_base_current.parameters()) + list(logits_layer_current.parameters()),
                                   lr=learning_rate)

        self.memory             = Memory()
        self.policy_function    = PolicyFunction(gamma, lam)  

        self.distributions      = Discrete()
        self.policy_loss        = RKHS_NPG(policy_params, vf_loss_coef, entropy_coef,
                                          gamma, lam)

        # Counter for outer iterations k (number of calls to update_npg)
        self.update_iter        = 0
        
        # Loss tracking
        self.td_error_losses    = []

        if is_training_mode:
          self.actor.train()
          self.critic.train()
        else:
          self.actor.eval()
          self.critic.eval()
          

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.save_eps(state, action, reward, done, next_state)

    def act(self, state):
        state           = torch.FloatTensor(state).unsqueeze(0).to(device).detach()
        action_probs    = self.actor(state)
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action  = self.distributions.sample(action_probs) 
        else:
            action  = torch.argmax(action_probs, 1)  
              
        return action.int().cpu().item()

    # Get loss and Do backpropagation
    def training_npg(self, states, actions, rewards, dones, next_states):
        action_probs = self.actor(states)
        q_values      = self.critic(states)            # (B,A)

        old_action_probs = self.actor_old(states)
        q_values_old     = self.critic_old(states)     # (B,A)

        next_q_values = self.critic(next_states)       # (B,A)

        loss, td_error_loss = self.policy_loss.compute_loss(
            action_probs,
            old_action_probs,
            q_values,
            q_values_old,
            next_q_values,
            actions,
            rewards,
            dones
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        
        return td_error_loss

    # Update the model
    def update_npg(self):        
        dataloader  = DataLoader(self.memory, self.batchsize, shuffle=False)
        batch_td_errors = []

        # -------- Optimize policy for T epochs --------
        for _ in range(self.T):
            for states, actions, rewards, dones, next_states in dataloader:
                td_loss = self.training_npg(states.float().to(device),
                                  actions.float().to(device),
                                  rewards.float().to(device),
                                  dones.float().to(device),
                                  next_states.float().to(device))
                batch_td_errors.append(td_loss)

        # After all epochs and batches, record the average TD error for this update step
        if batch_td_errors:
            avg_td_error = sum(batch_td_errors) / len(batch_td_errors)
            self.td_error_losses.append(avg_td_error)

        # -------- Deterministic penalty β schedule --------
        self.update_iter += 1  # k ← k + 1
        self.policy_params = float((self.update_iter) ** self.schedule_pow)  # β_k schedule
        self.policy_loss.policy_params = self.policy_params
        # print(f"[Schedule β] iteration={self.update_iter} → β set to {self.policy_params:.4f}")

        # -------- House-keeping --------
        self.memory.clear_memory()

        # Sync old networks with current
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def save_weights(self):
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, 'SlimeVolley/actor.tar')
        
        torch.save({
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, 'SlimeVolley/critic.tar')
        
    def load_weights(self):
        actor_checkpoint = torch.load('SlimeVolley/actor.tar')
        self.actor.load_state_dict(actor_checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])

        critic_checkpoint = torch.load('SlimeVolley/critic.tar')
        self.critic.load_state_dict(critic_checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])

class Runner():
    def __init__(self, env, agent, render, training_mode, n_update):
        self.env = env
        self.agent = agent
        self.render = render
        self.training_mode = training_mode

        self.n_update = n_update
        self.t_updates = 0

    def run_episode(self):
        ############################################
        state,_ = self.env.reset()    
        done = False
        total_reward = 0
        eps_time = 0
        ############################################
        for _ in range(10000):  # stpes
            action = self.agent.act(state)       
            next_state, reward, terminated, truncated, _ =  self.env.step(action)
            done = terminated or truncated
            eps_time += 1 
            self.t_updates += 1
            total_reward += reward
            
            if self.training_mode: 
                self.agent.save_eps(state.tolist(), action, reward, float(done), next_state.tolist()) 
                
            state = next_state
                    
            if self.render:
                self.env.render()     
            
            if self.training_mode and self.n_update is not None and self.t_updates == self.n_update:
                # update every n_update steps
                self.agent.update_npg()
                self.t_updates = 0
                
            if done: 
                break                
        
        if self.training_mode and self.n_update is None:
            self.agent.update_npg()
                    
        return total_reward, eps_time

def main():
    ############## Hyperparameters ##############
    load_weights        = False # If you want to load the agent, set this to True
    save_weights        = False # If you want to save the agent, set this to True
    training_mode       = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold    = 500 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

    render              = False # If you want to display the image, set this to True. Turn this off if you run this in Google Collab
    n_update            = 128 # How many episode before you update the Policy. Recommended set to 128 for Discrete
    n_episode           = 1000 # How many episode you want to run

    policy_params       = 20 # β will be overwritten by schedule
    entropy_coef        = 0.05 # How much randomness of action you will get
    vf_loss_coef        = 1.0 # Just set to 1
    batchsize           = 32 # How many batch per update. size of batch = n_update / minibatch. Recommended set to 4 for Discrete
    T                   = 4 # How many epoch per update. T in Algorithm 1.
    
    gamma               = 0.99
    lam                 = 0.95
    learning_rate       = 1e-3

    env_name            = 'CartPole-v1' # Set the env you want
    env                 = gym.make(env_name)
    state_dim           = env.observation_space.shape[0]
    action_dim          = env.action_space.n

    # --------- Experiment with different β schedules ----------
    schedule_pows = [-0.2, -0.5, -1.5]
    seeds         = [1,2,4,5,6,7,8,9,777,999]
    rewards_dict  = {}          # key=str(pow) -> np.ndarray shape (n_seeds, n_episode)
    td_error_dict = {}          # key=str(pow) -> list of td_error_losses per seed

    for pow_exp in schedule_pows:
        print(f"\n=== β schedule: k^{pow_exp} ===")
        runs_rewards = []
        runs_td_errors = []
        for sd in seeds:
            print(f"  Seed {sd}")
            np.random.seed(sd); torch.manual_seed(sd)
            env_i = gym.make(env_name)
            try:
                env_i.reset(seed=sd)
            except TypeError:
                pass  # older Gym
            if hasattr(env_i.action_space, 'seed'):
                env_i.action_space.seed(sd)

            agent_i = Agent(state_dim, action_dim, training_mode, policy_params,
                            entropy_coef, vf_loss_coef,
                            batchsize, T, gamma, lam, learning_rate,
                            schedule_pow=pow_exp)
            runner_i = Runner(env_i, agent_i, render, training_mode, n_update)

            ep_rewards = []
            for i_episode in range(1, n_episode + 1):
                total_reward, _ = runner_i.run_episode()
                ep_rewards.append(total_reward)
                if (i_episode % 100 == 0):
                    print(f"Pow {pow_exp} | Episode {i_episode} | Reward {total_reward}")
            runs_rewards.append(ep_rewards)
            runs_td_errors.append(agent_i.td_error_losses.copy())  # save TD error loss
            env_i.close()

        rewards_dict[str(pow_exp)] = np.array(runs_rewards)  # shape (n_seeds, n_episode)
        td_error_dict[str(pow_exp)] = runs_td_errors  # list of td_error_losses per seed

    # ---------- Plotting ----------
    plt.rcParams.update({'font.size': 24})
    plt.figure(figsize=(12,8))
    for pow_exp, rewards_mat in rewards_dict.items():
        mean_curve = rewards_mat.mean(axis=0)
        std_curve  = rewards_mat.std(axis=0)
        episodes   = np.arange(1, n_episode+1)
        plt.plot(episodes, mean_curve, label=f"k^{pow_exp}", linewidth=2)
        plt.fill_between(episodes, mean_curve-std_curve, mean_curve+std_curve, alpha=0.2)

    plt.xlabel("Episode", fontsize=24)
    plt.ylabel("Returns", fontsize=24)
    plt.title(f"Step Size Schedule Comparison on {env_name}", fontsize=24, fontweight='bold')
    plt.legend(fontsize=24)
    plt.grid(True)
    plt.tight_layout()
    
    plt.rcParams.update({'xtick.labelsize': 24, 'ytick.labelsize': 24})

    # ensure output directory exists
    import os
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("results", env_name, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Results will be saved to: {out_dir}")

    # Save all parameters to a JSON file
    import json
    all_params = {
        'load_weights': load_weights,
        'save_weights': save_weights,
        'training_mode': training_mode,
        'reward_threshold': reward_threshold,
        'render': render,
        'n_update': n_update,
        'n_episode': n_episode,
        'policy_params': policy_params,
        'entropy_coef': entropy_coef,
        'vf_loss_coef': vf_loss_coef,
        'batchsize': batchsize,
        'T': T,
        'gamma': gamma,
        'lam': lam,
        'learning_rate': learning_rate,
        'env_name': env_name,
        'schedule_pows': schedule_pows,
        'seeds': seeds,
    }
    params_path = os.path.join(out_dir, "parameters.json")
    with open(params_path, 'w') as f:
        json.dump(all_params, f, indent=4)

    # Save as PNG
    fig_path_png = os.path.join(out_dir, f"{env_name}_beta_schedule_raw_returns_lr{learning_rate}_seeds{len(seeds)}.png".replace('.', 'p'))
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    
    # Save as PDF
    fig_path_pdf = os.path.join(out_dir, f"{env_name}_beta_schedule_raw_returns_lr{learning_rate}_seeds{len(seeds)}.pdf")
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    
    plt.show()
    print(f"Raw returns plot saved to:")
    print(f"  - PNG: {fig_path_png}")
    print(f"  - PDF: {fig_path_pdf}")
    

    # -------- Moving-window (smoothed) return plots for various windows --------
    def moving_avg(arr, w):
        c = np.concatenate([[0], np.cumsum(arr)])
        return (c[w:] - c[:-w]) / w

    for window in [20, 40, 60, 80, 100]:
        plt.figure(figsize=(12,8))
        episodes_win = np.arange(window, n_episode+1)

        # NPG curves
        for pow_exp, rewards_mat in rewards_dict.items():
            # Ensure episode rewards are long enough for the window
            if rewards_mat.shape[1] < window:
                continue
            mv_curves = np.array([moving_avg(r, window) for r in rewards_mat])
            mean_mv = mv_curves.mean(axis=0)
            std_mv  = mv_curves.std(axis=0)
            plt.plot(episodes_win, mean_mv, label=f"k^{pow_exp}", linewidth=3)
            plt.fill_between(episodes_win, mean_mv-std_mv, mean_mv+std_mv, alpha=0.2)

        plt.xlabel("Episode", fontsize=24)
        plt.ylabel(f"Smoothed Returns (window={window})", fontsize=24)
        plt.title(f"Step Size Schedule Comparison on {env_name}", fontsize=24, fontweight='bold')
        plt.legend(fontsize=24)
        plt.grid(True)
        plt.tight_layout()
        
        plt.rcParams.update({'xtick.labelsize': 24, 'ytick.labelsize': 24})

        # Save as PNG
        fig_path_mv_png = os.path.join(out_dir, f"{env_name}_beta_schedule_smoothed_returns_w{window}_lr{learning_rate}_seeds{len(seeds)}.png".replace('.', 'p'))
        plt.savefig(fig_path_mv_png, dpi=300, bbox_inches='tight')
        
        # Save as PDF
        fig_path_mv_pdf = os.path.join(out_dir, f"{env_name}_beta_schedule_smoothed_returns_w{window}_lr{learning_rate}_seeds{len(seeds)}.pdf")
        plt.savefig(fig_path_mv_pdf, bbox_inches='tight')
        
        plt.show()
        print(f"Smoothed returns plot (w={window}) saved to:")
        print(f"  - PNG: {fig_path_mv_png}")
        print(f"  - PDF: {fig_path_mv_pdf}")


if __name__ == '__main__':
    main()