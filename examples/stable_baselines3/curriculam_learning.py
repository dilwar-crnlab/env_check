"""
curriculum_dqn_training.py

DQN training with curriculum learning for C+L band RMSA.
Implements three-stage curriculum:
  Stage 1: Routing quality only
  Stage 2: Routing + Band selection
  Stage 3: Full reward (Routing + Band + Spectrum allocation)
"""

import os
import pickle
import numpy as np
import torch
import gym
from collections import defaultdict
from tqdm import tqdm

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure



from run_viz_browser import LiveSpectrumVisualizer




print("="*80)
print("Curriculum DQN Training for C+L Band DeepRMSA")
print("="*80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("="*80)


# =============================================================================
# DQN with Action Masking
# =============================================================================

class MaskedDQN(DQN):
    """DQN with action masking support"""
    
    def predict(self, observation, state=None, episode_start=None,
                deterministic: bool = False, action_mask=None):
        """Predict action with optional masking"""
        
        if not deterministic and np.random.rand() < self.exploration_rate:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    action = np.array([np.random.choice(valid_actions)])
                else:
                    action = np.array([self.action_space.n - 1])
            else:
                action = np.array([self.action_space.sample()])
            return action, state
        
        observation = np.array(observation).reshape((-1,) + self.observation_space.shape)
        observation = torch.as_tensor(observation).to(self.device)
        
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.as_tensor(action_mask).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_net(observation)
                masked_q = torch.where(
                    action_mask,
                    q_values,
                    torch.tensor(-1e8, device=q_values.device)
                )
                action = masked_q.argmax(dim=1).cpu().numpy()
        else:
            with torch.no_grad():
                q_values = self.q_net(observation)
                action = q_values.argmax(dim=1).cpu().numpy()
        
        return action, state


# =============================================================================
# Action Masking
# =============================================================================

def get_action_mask(env):
    """Get boolean mask of valid actions"""
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    
    total_actions = env.action_space.n
    mask = np.zeros(total_actions, dtype=bool)
    
    src = base_env.current_service.source
    dst = base_env.current_service.destination
    paths = base_env.k_shortest_paths[(src, dst)]
    
    for path_idx in range(base_env.k_paths):
        path = paths[path_idx]
        
        for band in range(base_env.num_bands):
            initial_indices, lengths = base_env.get_available_blocks(
                path_idx, base_env.num_bands, band, base_env.modulations
            )
            
            required_slots = base_env.get_number_slots(
                path, base_env.num_bands, band, base_env.modulations
            )
            
            for block_idx in range(len(initial_indices)):
                if block_idx < base_env.j and lengths[block_idx] >= required_slots:
                    action = (path_idx + 
                             (band * base_env.k_paths) + 
                             (block_idx * base_env.k_paths * base_env.num_bands))
                    if action < total_actions:
                        mask[action] = True
    
    if base_env.reject_action == 1:
        mask[-1] = True
    
    return mask


# =============================================================================
# Curriculum Training Loop
# =============================================================================

def train_curriculum_dqn(env, num_episodes=10000, log_dir='./logs/curriculum_dqn/',
                         device='cuda', save_freq=500, print_freq=50):
    
    os.makedirs(log_dir, exist_ok=True)


    visualizer = LiveSpectrumVisualizer(env, port=5002)
    visualizer.start(blocking=False)
    
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    
    if not hasattr(base_env, 'use_curriculum') or not base_env.use_curriculum:
        print("\n" + "!"*80)
        print("WARNING: Curriculum learning is not enabled in environment!")
        print("!"*80 + "\n")
    
    print(f"\n{'='*80}")
    print(f"{'Curriculum Configuration':^80}")
    print(f"{'='*80}")
    print(f"Mode: {'Adaptive' if base_env.adaptive_curriculum else 'Fixed Episodes'}")
    print(f"\nStage Thresholds:")
    print(f"  Stage 1→2: Episode {base_env.curriculum_stage_1_episodes:,}")
    print(f"  Stage 2→3: Episode {base_env.curriculum_stage_2_episodes:,}")
    print(f"\nReward Weights:")
    print(f"  Routing (w_r):  {base_env.w_routing}")
    print(f"  Band (w_b):     {base_env.w_band}")
    print(f"  Spectrum (w_s): {base_env.w_spectrum}")
    print(f"\nClassification:")
    print(f"  ML_avg: {base_env.ml_avg}")
    print(f"  F_avg:  {base_env.f_avg} slots")
    print(f"{'='*80}\n")
    
    agent = MaskedDQN(
        "MlpPolicy",
        env,
        verbose=0,
        device=device,
        learning_rate=1e-5,
        buffer_size=100000,
        learning_starts=2000,
        batch_size=64,
        tau=1.0,
        gamma=0.95,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128, 128, 128, 128]),
        tensorboard_log=log_dir
    )
    
    agent.set_logger(configure(folder=log_dir, format_strings=['stdout', 'csv']))
    
    print(f"Action space: {env.action_space.n}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"\nTraining for {num_episodes:,} episodes...")
    print("="*80 + "\n")
    
    metrics = defaultdict(list)
    blocking_reasons_history = []
    last_stage = 1
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        obs = env.reset()
        visualizer.update()
        done = False
        episode_reward = 0
        steps = 0
        valid_actions = 0
        invalid_actions = 0
        
        current_stage = base_env._get_curriculum_stage()
        
        if current_stage != last_stage:
            print(f"\n{'='*80}")
            print(f"CURRICULUM STAGE TRANSITION: {last_stage} → {current_stage}")
            print(f"Episode: {episode + 1}")
            print(f"{'='*80}\n")
            
            transition_path = os.path.join(log_dir, f'stage_{current_stage}_ep{episode+1}')
            agent.save(transition_path)
            print(f"Model saved at stage transition: {transition_path}\n")
            last_stage = current_stage
        
        while not done:
            mask = get_action_mask(env)
            action, _ = agent.predict(obs, deterministic=False, action_mask=mask)
            
            if isinstance(action, np.ndarray):
                action = action.item()
            
            if mask[action]:
                valid_actions += 1
            else:
                invalid_actions += 1
            
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            visualizer.update()  # Update after each step
        
        # Store metrics
        metrics['episode'].append(episode + 1)
        metrics['stage'].append(current_stage)
        metrics['reward'].append(episode_reward)
        metrics['steps'].append(steps)
        metrics['blocking_rate'].append(info.get('episode_service_blocking_rate', 0))
        metrics['network_frag'].append(info.get('network_fragmentation', 0))
        metrics['valid_actions'].append(valid_actions)
        metrics['invalid_actions'].append(invalid_actions)
        
        if 'blocking_reasons_count' in info:
            blocking_reasons_history.append(info['blocking_reasons_count'].copy())
        
        if 'band_fragmentation' in info:
            band_frag = info['band_fragmentation']
            metrics['c_band_frag'].append(band_frag.get(0, 0.0))
            if base_env.num_bands > 1:
                metrics['l_band_frag'].append(band_frag.get(1, 0.0))
        
        # FIX: Correct band utilization calculation
        available_slots_array = base_env.topology.graph.get('available_slots', None)
        if available_slots_array is not None:
            num_edges = base_env.topology.number_of_edges()
            
            # C-band: first num_edges rows
            c_start, c_end = base_env.get_shift(0)
            c_band_rows = slice(0, num_edges)
            c_band_slots = (c_end - c_start) * num_edges
            c_free = np.sum(available_slots_array[c_band_rows, c_start:c_end])
            c_util = (c_band_slots - c_free) / c_band_slots if c_band_slots > 0 else 0.0
            metrics['c_band_util'].append(c_util)
            
            # L-band: next num_edges rows
            if base_env.num_bands > 1:
                l_start, l_end = base_env.get_shift(1)
                l_band_rows = slice(num_edges, 2 * num_edges)
                l_band_slots = (l_end - l_start) * num_edges
                l_free = np.sum(available_slots_array[l_band_rows, l_start:l_end])
                l_util = (l_band_slots - l_free) / l_band_slots if l_band_slots > 0 else 0.0
                metrics['l_band_util'].append(l_util)
        
        # OSNR margin
        running_services = base_env.topology.graph.get("running_services", [])
        if running_services:
            margins = [s.OSNR_margin for s in running_services if hasattr(s, 'OSNR_margin')]
            avg_osnr = np.mean(margins) if margins else 0.0
        else:
            avg_osnr = 0.0
        metrics['avg_osnr_margin'].append(avg_osnr)
        
        # Print statistics
        if (episode + 1) % print_freq == 0 and (episode + 1) >= print_freq:
            curriculum_stats = base_env.get_curriculum_statistics(last_n=500)
            
            if 'error' not in curriculum_stats:
                recent_window = min(print_freq, episode + 1)
                
                print(f"\n{'='*80}")
                print(f"Episode {episode + 1:,}/{num_episodes:,} | "
                      f"Stage {curriculum_stats['current_stage']}/3 | "
                      f"Exploration: {agent.exploration_rate:.1%}")
                print(f"{'='*80}")
                
                print(f"\nPerformance (last {recent_window} episodes):")
                print(f"  Reward:         {np.mean(metrics['reward'][-recent_window:]):>8.2f}")
                print(f"  Blocking Rate:  {np.mean(metrics['blocking_rate'][-recent_window:])*100:>7.2f}%")
                print(f"  Network Frag:   {np.mean(metrics['network_frag'][-recent_window:]):>8.3f}")
                print(f"  OSNR Margin:    {np.mean(metrics['avg_osnr_margin'][-recent_window:]):>7.2f} dB")
                
                print(f"\nBand Utilization:")
                print(f"  C-band:         {np.mean(metrics['c_band_util'][-recent_window:])*100:>7.2f}%")
                if base_env.num_bands > 1:
                    print(f"  L-band:         {np.mean(metrics['l_band_util'][-recent_window:])*100:>7.2f}%")
                
                print(f"\nCurriculum Progress:")
                print(f"  Routing Mastery:    {curriculum_stats['routing_mastery_%']:>6.1f}% "
                      f"(avg: {curriculum_stats['avg_routing_bonus']:.3f})")
                print(f"  Band Accuracy:      {curriculum_stats['band_accuracy']*100:>6.1f}% "
                      f"(correct assignments)")
                print(f"  Spectrum Quality:   {curriculum_stats['avg_spectrum_bonus']:>6.3f} "
                      f"(FF/LF adherence)")
                print(f"  Acceptance Rate:    {curriculum_stats['acceptance_rate']*100:>6.1f}%")
                
                if len(blocking_reasons_history) >= recent_window:
                    recent_reasons = blocking_reasons_history[-recent_window:]
                    agg_reasons = defaultdict(int)
                    total_blocked = 0
                    
                    for ep_reasons in recent_reasons:
                        for reason, count in ep_reasons.items():
                            if 'total' in reason:
                                total_blocked += count
                            else:
                                agg_reasons[reason] += count
                    
                    if total_blocked > 0:
                        print(f"\nBlocking Reasons (last {recent_window} episodes):")
                        sorted_reasons = sorted(agg_reasons.items(), 
                                              key=lambda x: x[1], reverse=True)
                        
                        for reason, count in sorted_reasons[:5]:
                            if count > 0:
                                pct = (count / total_blocked * 100)
                                print(f"  {reason:28s}: {count:>5d} ({pct:>5.1f}%)")
                
                recent_valid = np.sum(metrics['valid_actions'][-recent_window:])
                recent_invalid = np.sum(metrics['invalid_actions'][-recent_window:])
                total_actions = recent_valid + recent_invalid
                if total_actions > 0:
                    print(f"\nAction Validity:")
                    print(f"  Valid:   {recent_valid:>6d} ({recent_valid/total_actions*100:>5.1f}%)")
                    print(f"  Invalid: {recent_invalid:>6d} ({recent_invalid/total_actions*100:>5.1f}%)")
        
        if (episode + 1) % save_freq == 0:
            save_path = os.path.join(log_dir, f'model_ep{episode+1}')
            agent.save(save_path)
            print(f"\n{'─'*80}")
            print(f"Model checkpoint saved: episode {episode+1}")
            print(f"{'─'*80}\n")
    
    print(f"\n{'='*80}")
    print(f"{'TRAINING COMPLETE':^80}")
    print(f"{'='*80}\n")
    
    final_window = min(1000, num_episodes)
    final_stats = base_env.get_curriculum_statistics(last_n=final_window)
    
    print(f"Final Performance (last {final_window} episodes):")
    print(f"{'─'*80}")
    print(f"  Average Reward:        {np.mean(metrics['reward'][-final_window:]):>8.2f}")
    print(f"  Blocking Rate:         {np.mean(metrics['blocking_rate'][-final_window:])*100:>7.2f}%")
    print(f"  Network Fragmentation: {np.mean(metrics['network_frag'][-final_window:]):>8.3f}")
    print(f"  OSNR Margin:           {np.mean(metrics['avg_osnr_margin'][-final_window:]):>7.2f} dB")
    print(f"\nBand Utilization:")
    print(f"  C-band:                {np.mean(metrics['c_band_util'][-final_window:])*100:>7.2f}%")
    if base_env.num_bands > 1:
        print(f"  L-band:                {np.mean(metrics['l_band_util'][-final_window:])*100:>7.2f}%")
    
    if 'error' not in final_stats:
        print(f"\nCurriculum Metrics:")
        print(f"  Final Stage:           {final_stats['current_stage']}/3")
        print(f"  Routing Mastery:       {final_stats['routing_mastery_%']:>6.1f}%")
        print(f"  Band Accuracy:         {final_stats['band_accuracy']*100:>6.1f}%")
        print(f"  Spectrum Quality:      {final_stats['avg_spectrum_bonus']:>6.3f}")
        print(f"  Acceptance Rate:       {final_stats['acceptance_rate']*100:>6.1f}%")
    
    print(f"{'='*80}\n")
    
    final_path = os.path.join(log_dir, 'final_model')
    agent.save(final_path)
    print(f"Final model saved: {final_path}")
    
    metrics_path = os.path.join(log_dir, 'training_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(dict(metrics), f)
    print(f"Metrics saved: {metrics_path}\n")
    
    return metrics, agent


# =============================================================================
# Main
# =============================================================================

def main():
    topology_name = 'indian_net'
    k_paths = 5
    topology_path = f'../topologies/{topology_name}_{k_paths}-paths_new.h5'
    
    print(f"\nLoading topology: {topology_path}")
    
    if not os.path.exists(topology_path):
        print(f"ERROR: Topology not found: {topology_path}")
        return
    
    with open(topology_path, 'rb') as f:
        topology = pickle.load(f)
    
    print(f"Loaded {topology_name}: {topology.number_of_nodes()} nodes, "
          f"{topology.number_of_edges()} edges\n")
    
    env_args = dict(
        topology=topology,
        seed=42,
        allow_rejection=False,
        j=1,
        mean_service_holding_time=200.0,
        mean_service_inter_arrival_time=1.0,
        episode_length=100,
        num_bands=2,
        bit_rates=[50, 100, 200]
    )
    
    print("Creating environment...")
    env = gym.make('DeepRMSA-v0', **env_args)
    
    print("Environment created\n")
    
    log_dir = './logs/curriculum_dqn/'
    num_episodes = 10000
    
    metrics, agent = train_curriculum_dqn(
        env=env,
        num_episodes=num_episodes,
        log_dir=log_dir,
        device=device,
        save_freq=500,
        print_freq=50
    )
    
    print(f"\n{'='*80}")
    print("All done!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()