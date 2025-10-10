"""
Integration of Live Spectrum Visualization with Parallel Training
Adds browser-based real-time visualization alongside matplotlib plots
"""

import os
import pickle
import time
from collections import defaultdict
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import queue

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
import gym
import torch

# Import visualizer and agent
from run_viz_browser import LiveSpectrumVisualizer
from ksp_ff_osnr_check import KSP_FF_Agent

# ... [Keep all your existing class definitions: PathBandEnv, BlockEnv, MetricsWrapper, etc.]

from H_DQN_KSP_FF import MetricsWrapper, PathBandEnv, BlockEnv, HierarchicalCoordinator, plot_parallel_metrics

# =============================================================================
# Modified Training Functions with Visualization Support
# =============================================================================

# def train_hierarchical_dqn_with_viz(env_args, log_dir, metrics_queue, viz_queue, stop_event, device, num_episodes):
#     """Train hierarchical DQN with visualization updates"""
    
#     base_env = gym.make('DeepRMSA-v0', **env_args)
#     metrics_env = MetricsWrapper(base_env)
#     path_band_env = PathBandEnv(metrics_env)
#     block_env = BlockEnv(metrics_env, path_band_env)
    
#     # Get base env for visualization data
#     base_for_viz = metrics_env
#     while hasattr(base_for_viz, 'env'):
#         base_for_viz = base_for_viz.env
    
#     coordinator = HierarchicalCoordinator(path_band_env, block_env, device)
    
#     pbar = tqdm(total=num_episodes, desc="[H-DQN Training]", 
#                 position=0, leave=True, dynamic_ncols=True)
    
#     episode = 0
#     step_count = 0
    
#     while episode < num_episodes and not stop_event.is_set():
#         episode_reward, info = coordinator.train_episode()
#         episode += 1
#         step_count += info.get('episode_length', 100)
        
#         pbar.update(1)
#         pbar.set_postfix({
#             'Blocking': f"{info.get('episode_service_blocking_rate', 0):.3f}",
#             'Frag': f"{info.get('network_fragmentation', 0):.2f}",
#             'Bitrate': f"{info.get('episode_bitrate_served_gbps', 0):.1f}",
#             'Masked': info.get('masked_invalid_blocks', 0)
#         })
        
#         # Send metrics for matplotlib plotting
#         metrics = {
#             'algorithm': 'H-DQN',
#             'episode': episode,
#             'blocking_rate': info.get('episode_service_blocking_rate', 0),
#             'bit_rate_blocking': info.get('episode_bit_rate_blocking_rate', 0),
#             'network_frag': info.get('network_fragmentation', 0),
#             'bitrate_served': info.get('episode_bitrate_served_gbps', 0),
#             'masked_invalids': info.get('masked_invalid_blocks', 0),
#         }
        
#         try:
#             metrics_queue.put(metrics, block=False)
#         except queue.Full:
#             pass
        
#         # Send visualization data every N steps
#         viz_data = extract_viz_data(base_for_viz, episode, 'H-DQN', info)
#         try:
#             viz_queue.put(viz_data, block=False)
#         except queue.Full:
#             pass
    
#     pbar.close()
#     coordinator.save(os.path.join(log_dir, 'final_model'))
#     stop_event.set()
#     print(f"\n[H-DQN] Complete ({episode} episodes)")

def train_hierarchical_dqn_with_viz(env_args, log_dir, metrics_queue, viz_queue, stop_event, device, num_episodes):
    """Train hierarchical DQN with visualization updates"""
    
    base_env = gym.make('DeepRMSA-v0', **env_args)
    metrics_env = MetricsWrapper(base_env)
    path_band_env = PathBandEnv(metrics_env)
    block_env = BlockEnv(metrics_env, path_band_env)
    
    # Get base env for visualization data
    base_for_viz = metrics_env
    while hasattr(base_for_viz, 'env'):
        base_for_viz = base_for_viz.env
    
    coordinator = HierarchicalCoordinator(path_band_env, block_env, device)
    
    pbar = tqdm(total=num_episodes, desc="[H-DQN Training]", 
                position=0, leave=True, dynamic_ncols=True)
    
    episode = 0
    
    while episode < num_episodes and not stop_event.is_set():
        episode_reward, info = coordinator.train_episode()
        episode += 1
        
        pbar.update(1)
        pbar.set_postfix({
            'Blocking': f"{info.get('episode_service_blocking_rate', 0):.3f}",
            'Frag': f"{info.get('network_fragmentation', 0):.2f}",
            'Bitrate': f"{info.get('episode_bitrate_served_gbps', 0):.1f}",
            'Masked': info.get('masked_invalid_blocks', 0)
        })
        
        # Send metrics for matplotlib plotting
        metrics = {
            'algorithm': 'H-DQN',
            'episode': episode,
            'blocking_rate': info.get('episode_service_blocking_rate', 0),
            'bit_rate_blocking': info.get('episode_bit_rate_blocking_rate', 0),
            'network_frag': info.get('network_fragmentation', 0),
            'bitrate_served': info.get('episode_bitrate_served_gbps', 0),
            'masked_invalids': info.get('masked_invalid_blocks', 0),
        }
        
        try:
            metrics_queue.put(metrics, block=False)
        except queue.Full:
            pass
        
        # Send enhanced visualization data for browser
        viz_data = extract_viz_data(base_for_viz, episode, 'H-DQN', info)
        try:
            viz_queue.put(viz_data, block=False)
        except queue.Full:
            pass
    
    pbar.close()
    coordinator.save(os.path.join(log_dir, 'final_model'))
    stop_event.set()
    print(f"\n[H-DQN] Complete ({episode} episodes)")



# def run_kspff_parallel_with_viz(env_args, log_dir, metrics_queue, viz_queue, stop_event, num_episodes):
#     """Run KSP-FF with visualization updates"""
    
#     base_env = gym.make('DeepRMSA-v0', **env_args)
#     env = MetricsWrapper(base_env)
    
#     base_for_ksp = env
#     while hasattr(base_for_ksp, 'env'):
#         base_for_ksp = base_for_ksp.env
    
#     agent = KSP_FF_Agent(base_for_ksp, debug=False)
    
#     pbar = tqdm(total=num_episodes, desc="[KSP-FF Baseline]", 
#                 position=1, leave=True, dynamic_ncols=True)
    
#     episode = 0
#     step_count = 0
    
#     while episode < num_episodes and not stop_event.is_set():
#         obs = env.reset()
#         done = False
        
#         while not done:
#             path_idx, band_idx, initial_slot = agent.select_action(obs)
            
#             # Convert to DeepRMSA action
#             if path_idx < base_for_ksp.k_paths and band_idx < base_for_ksp.num_bands:
#                 initial_indices, _ = base_for_ksp.get_available_blocks(
#                     path_idx, base_for_ksp.num_bands, band_idx, base_for_ksp.modulations
#                 )
                
#                 if initial_slot in initial_indices:
#                     block_idx = list(initial_indices).index(initial_slot)
#                     action = path_idx + (band_idx * base_for_ksp.k_paths) + \
#                             (block_idx * base_for_ksp.k_paths * base_for_ksp.num_bands)
#                 else:
#                     action = base_for_ksp.k_paths * base_for_ksp.num_bands * base_for_ksp.j
#             else:
#                 action = base_for_ksp.k_paths * base_for_ksp.num_bands * base_for_ksp.j
            
#             obs, reward, done, info = env.step(action)
#             agent.update_stats((path_idx, band_idx, initial_slot), reward, info)
#             step_count += 1
            
#             # Send visualization data
#             # if step_count % 10 == 0:
#             #     try:
#             #         viz_data = extract_viz_data(base_for_ksp, step_count, 'KSP-FF')
#             #         viz_queue.put(viz_data, block=False)
#             #     except queue.Full:
#             #         pass

#             viz_data = extract_viz_data(base_for_ksp, episode, 'H-DQN', info)
#             try:
#                 viz_queue.put(viz_data, block=False)
#             except queue.Full:
#                 pass
        
#         episode += 1
        
#         pbar.update(1)
#         pbar.set_postfix({
#             'Blocking': f"{info.get('episode_service_blocking_rate', 0):.3f}",
#             'Frag': f"{info.get('network_fragmentation', 0):.2f}",
#             'Bitrate': f"{info.get('episode_bitrate_served_gbps', 0):.1f}"
#         })
        
#         metrics = {
#             'algorithm': 'KSP-FF',
#             'episode': episode,
#             'blocking_rate': info.get('episode_service_blocking_rate', 0),
#             'bit_rate_blocking': info.get('episode_bit_rate_blocking_rate', 0),
#             'network_frag': info.get('network_fragmentation', 0),
#             'bitrate_served': info.get('episode_bitrate_served_gbps', 0),
#             'masked_invalids': 0,
#         }
        
#         try:
#             metrics_queue.put(metrics, block=False)
#         except queue.Full:
#             pass
    
#     pbar.close()
#     stats = agent.get_statistics()
#     print(f"\n[KSP-FF] Complete ({episode} episodes)")
#     print(f"  Acceptance Rate: {stats.get('acceptance_rate', 0):.4f}")

def run_kspff_parallel_with_viz(env_args, log_dir, metrics_queue, viz_queue, stop_event, num_episodes):
    """Run KSP-FF with visualization updates"""
    
    base_env = gym.make('DeepRMSA-v0', **env_args)
    env = MetricsWrapper(base_env)
    
    base_for_ksp = env
    while hasattr(base_for_ksp, 'env'):
        base_for_ksp = base_for_ksp.env
    
    agent = KSP_FF_Agent(base_for_ksp, debug=False)
    
    pbar = tqdm(total=num_episodes, desc="[KSP-FF Baseline]", 
                position=1, leave=True, dynamic_ncols=True)
    
    episode = 0
    
    while episode < num_episodes and not stop_event.is_set():
        obs = env.reset()
        done = False
        
        while not done:
            path_idx, band_idx, initial_slot = agent.select_action(obs)
            
            # Convert to DeepRMSA action
            if path_idx < base_for_ksp.k_paths and band_idx < base_for_ksp.num_bands:
                initial_indices, _ = base_for_ksp.get_available_blocks(
                    path_idx, base_for_ksp.num_bands, band_idx, base_for_ksp.modulations
                )
                
                if initial_slot in initial_indices:
                    block_idx = list(initial_indices).index(initial_slot)
                    action = path_idx + (band_idx * base_for_ksp.k_paths) + \
                            (block_idx * base_for_ksp.k_paths * base_for_ksp.num_bands)
                else:
                    action = base_for_ksp.k_paths * base_for_ksp.num_bands * base_for_ksp.j
            else:
                action = base_for_ksp.k_paths * base_for_ksp.num_bands * base_for_ksp.j
            
            obs, reward, done, info = env.step(action)
            agent.update_stats((path_idx, band_idx, initial_slot), reward, info)
        
        episode += 1
        
        pbar.update(1)
        pbar.set_postfix({
            'Blocking': f"{info.get('episode_service_blocking_rate', 0):.3f}",
            'Frag': f"{info.get('network_fragmentation', 0):.2f}",
            'Bitrate': f"{info.get('episode_bitrate_served_gbps', 0):.1f}"
        })
        
        # Send metrics for matplotlib
        metrics = {
            'algorithm': 'KSP-FF',
            'episode': episode,
            'blocking_rate': info.get('episode_service_blocking_rate', 0),
            'bit_rate_blocking': info.get('episode_bit_rate_blocking_rate', 0),
            'network_frag': info.get('network_fragmentation', 0),
            'bitrate_served': info.get('episode_bitrate_served_gbps', 0),
            'masked_invalids': 0,
        }
        
        try:
            metrics_queue.put(metrics, block=False)
        except queue.Full:
            pass
        
        # Send visualization data for browser
        viz_data = extract_viz_data(base_for_ksp, episode, 'KSP-FF', info)
        try:
            viz_queue.put(viz_data, block=False)
        except queue.Full:
            pass
    
    pbar.close()
    stats = agent.get_statistics()
    print(f"\n[KSP-FF] Complete ({episode} episodes)")
    print(f"  Acceptance Rate: {stats.get('acceptance_rate', 0):.4f}")


# def extract_viz_data(env, step_count, algorithm):
#     """Extract visualization data from environment"""
#     available_slots = env.topology.graph.get('available_slots')
    
#     if available_slots is None:
#         return None
    
#     # Extract spectrum data for each band
#     bands_data = []
#     for band_idx in range(env.num_bands):
#         start, end = env.get_shift(band_idx)
#         band_spectrum = []
        
#         for edge_idx in range(env.topology.number_of_edges()):
#             offset = edge_idx + (env.topology.number_of_edges() * band_idx)
#             link_slots = (1 - available_slots[offset, start:end]).tolist()
#             band_spectrum.append(link_slots)
        
#         total_slots = env.topology.number_of_edges() * (end - start)
#         occupied = np.sum(1 - available_slots[
#             band_idx * env.topology.number_of_edges():(band_idx + 1) * env.topology.number_of_edges(),
#             start:end
#         ])
#         utilization = float(occupied / total_slots * 100) if total_slots > 0 else 0.0
        
#         band_frag = env.topology.graph.get("current_band_fragmentation", {}).get(band_idx, 0.0)
        
#         bands_data.append({
#             'index': band_idx,
#             'name': "C-band" if band_idx == 0 else "L-band",
#             'utilization': round(utilization, 2),
#             'fragmentation': round(band_frag, 2)
#         })
    
#     # Calculate metrics
#     running_services = env.topology.graph.get("running_services", [])
#     osnr_margins = [s.OSNR_margin for s in running_services if hasattr(s, 'OSNR_margin')]
#     avg_osnr = round(np.mean(osnr_margins), 2) if osnr_margins else 0.0
    
#     # Network load in Erlang
#     if hasattr(env, 'mean_service_holding_time') and hasattr(env, 'mean_service_inter_arrival_time'):
#         arrival_rate = 1.0 / env.mean_service_inter_arrival_time
#         holding_time = env.mean_service_holding_time
#         network_load = arrival_rate * holding_time
#     else:
#         network_load = float(len(running_services))
    
#     return {
#         'algorithm': algorithm,
#         'step': step_count,
#         'services_processed': env.services_processed,
#         'services_accepted': env.services_accepted,
#         'blocking_rate': round(
#             (env.services_processed - env.services_accepted) / env.services_processed * 100, 2
#         ) if env.services_processed > 0 else 0.0,
#         'network_load': round(network_load, 2),
#         'avg_osnr_margin': avg_osnr,
#         'band_stats': bands_data,
#     }
def extract_viz_data(env, episode, algorithm, info):
    """Extract comprehensive visualization data including all matplotlib metrics"""
    
    # Band statistics
    bands_data = []
    available_slots = env.topology.graph.get('available_slots')
    
    if available_slots is not None:
        for band_idx in range(env.num_bands):
            start, end = env.get_shift(band_idx)
            total_slots = env.topology.number_of_edges() * (end - start)
            occupied = np.sum(1 - available_slots[
                band_idx * env.topology.number_of_edges():(band_idx + 1) * env.topology.number_of_edges(),
                start:end
            ])
            utilization = float(occupied / total_slots * 100) if total_slots > 0 else 0.0
            band_frag = env.topology.graph.get("current_band_fragmentation", {}).get(band_idx, 0.0)
            
            bands_data.append({
                'index': band_idx,
                'name': "C-band" if band_idx == 0 else "L-band",
                'utilization': round(utilization, 2),
                'fragmentation': round(band_frag, 3)
            })
    
    # OSNR data
    running_services = env.topology.graph.get("running_services", [])
    osnr_margins = [s.OSNR_margin for s in running_services if hasattr(s, 'OSNR_margin')]
    avg_osnr = round(np.mean(osnr_margins), 2) if osnr_margins else 0.0
    
    # Network load
    if hasattr(env, 'mean_service_holding_time') and hasattr(env, 'mean_service_inter_arrival_time'):
        arrival_rate = 1.0 / env.mean_service_inter_arrival_time
        holding_time = env.mean_service_holding_time
        network_load = arrival_rate * holding_time
    else:
        network_load = float(len(running_services))
    
    return {
        'algorithm': algorithm,
        'episode': episode,
        'blocking_rate': info.get('episode_service_blocking_rate', 0) * 100,  # Convert to percentage
        'bit_rate_blocking': info.get('episode_bit_rate_blocking_rate', 0) * 100,
        'network_frag': info.get('network_fragmentation', 0),
        'bitrate_served': info.get('episode_bitrate_served_gbps', 0),
        'masked_invalids': info.get('masked_invalid_blocks', 0),
        'network_load': round(network_load, 2),
        'avg_osnr_margin': avg_osnr,
        'band_stats': bands_data,
    }


def run_browser_visualization(viz_queue, stop_event, port=5000):
    """Run browser visualization server consuming data from queue"""
    from flask import Flask, render_template_string
    from flask_socketio import SocketIO
    import threading
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'training-viz-secret'
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Store latest data
    latest_data = {'H-DQN': None, 'KSP-FF': None}
    
    @app.route('/')
    def index():
        return render_template_string(TRAINING_VIZ_HTML)
    
    @socketio.on('connect')
    def handle_connect():
        print(f"Browser client connected to training visualization")
    
    def consume_queue():
        """Background thread to consume visualization queue"""
        while not stop_event.is_set():
            try:
                viz_data = viz_queue.get(timeout=0.5)
                if viz_data:
                    algorithm = viz_data['algorithm']
                    latest_data[algorithm] = viz_data
                    # Send with the event name that the HTML expects
                    socketio.emit('metrics_update', viz_data)
            except queue.Empty:
                continue
        print("\n[Browser Viz] Queue consumer stopped")
    
    # Start queue consumer
    consumer_thread = threading.Thread(target=consume_queue, daemon=True)
    consumer_thread.start()
    
    print(f"\n{'='*60}")
    print(f"Training Visualization Server")
    print(f"{'='*60}")
    print(f"Open http://localhost:{port} in your browser")
    print(f"{'='*60}\n")
    
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False, 
                    use_reloader=False, log_output=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"[Browser Viz] Error: {e}")
# HTML Template with OSNR Chart
# TRAINING_VIZ_HTML = """
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Training Visualization</title>
#     <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
#     <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
#     <style>
#         * { margin: 0; padding: 0; box-sizing: border-box; }
#         body {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             background: #0a0a0a;
#             color: #ffffff;
#             padding: 15px;
#         }
#         .container { max-width: 1800px; margin: 0 auto; }
#         h1 {
#             text-align: center;
#             margin-bottom: 20px;
#             color: #4CAF50;
#             font-size: 2em;
#         }
#         .grid {
#             display: grid;
#             grid-template-columns: repeat(2, 1fr);
#             gap: 20px;
#             margin-bottom: 20px;
#         }
#         .algorithm-panel {
#             background: #1a1a1a;
#             border: 2px solid #333;
#             border-radius: 10px;
#             padding: 20px;
#         }
#         .algorithm-panel.hdqn { border-color: #2196F3; }
#         .algorithm-panel.kspff { border-color: #f44336; }
#         .panel-title {
#             font-size: 1.5em;
#             margin-bottom: 15px;
#             font-weight: bold;
#         }
#         .hdqn .panel-title { color: #2196F3; }
#         .kspff .panel-title { color: #f44336; }
#         .metrics-grid {
#             display: grid;
#             grid-template-columns: repeat(2, 1fr);
#             gap: 10px;
#             margin-bottom: 15px;
#         }
#         .metric {
#             background: #0a0a0a;
#             padding: 12px;
#             border-radius: 5px;
#             border: 1px solid #333;
#         }
#         .metric-label {
#             color: #888;
#             font-size: 0.85em;
#             margin-bottom: 5px;
#         }
#         .metric-value {
#             font-size: 1.3em;
#             font-weight: bold;
#             color: #4CAF50;
#         }
#         .band-stats {
#             display: flex;
#             gap: 10px;
#             margin-top: 10px;
#         }
#         .band-stat {
#             flex: 1;
#             background: #0a0a0a;
#             padding: 10px;
#             border-radius: 5px;
#             border: 1px solid #333;
#         }
#         .chart-container {
#             background: #1a1a1a;
#             border: 2px solid #333;
#             border-radius: 10px;
#             padding: 20px;
#             height: 400px;
#         }
#         .status {
#             text-align: center;
#             padding: 10px;
#             background: #1a1a1a;
#             border-radius: 5px;
#             margin-top: 15px;
#             color: #888;
#         }
#         .connected { color: #4CAF50; }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>🚀 Parallel Training: H-DQN vs KSP-FF</h1>
        
#         <div class="grid">
#             <div class="algorithm-panel hdqn">
#                 <div class="panel-title">Hierarchical DQN</div>
#                 <div class="metrics-grid">
#                     <div class="metric">
#                         <div class="metric-label">Step</div>
#                         <div class="metric-value" id="hdqn-step">0</div>
#                     </div>
#                     <div class="metric">
#                         <div class="metric-label">Blocking Rate</div>
#                         <div class="metric-value" id="hdqn-blocking">0%</div>
#                     </div>
#                     <div class="metric">
#                         <div class="metric-label">Network Load</div>
#                         <div class="metric-value" id="hdqn-load">0 E</div>
#                     </div>
#                     <div class="metric">
#                         <div class="metric-label">Avg OSNR Margin</div>
#                         <div class="metric-value" id="hdqn-osnr">0.0 dB</div>
#                     </div>
#                 </div>
#                 <div class="band-stats">
#                     <div class="band-stat">
#                         <div class="metric-label">C-band</div>
#                         <div>Util: <span id="hdqn-c-util">0</span>%</div>
#                         <div>Frag: <span id="hdqn-c-frag">0</span></div>
#                     </div>
#                     <div class="band-stat">
#                         <div class="metric-label">L-band</div>
#                         <div>Util: <span id="hdqn-l-util">0</span>%</div>
#                         <div>Frag: <span id="hdqn-l-frag">0</span></div>
#                     </div>
#                 </div>
#             </div>
            
#             <div class="algorithm-panel kspff">
#                 <div class="panel-title">KSP-FF Baseline</div>
#                 <div class="metrics-grid">
#                     <div class="metric">
#                         <div class="metric-label">Step</div>
#                         <div class="metric-value" id="kspff-step">0</div>
#                     </div>
#                     <div class="metric">
#                         <div class="metric-label">Blocking Rate</div>
#                         <div class="metric-value" id="kspff-blocking">0%</div>
#                     </div>
#                     <div class="metric">
#                         <div class="metric-label">Network Load</div>
#                         <div class="metric-value" id="kspff-load">0 E</div>
#                     </div>
#                     <div class="metric">
#                         <div class="metric-label">Avg OSNR Margin</div>
#                         <div class="metric-value" id="kspff-osnr">0.0 dB</div>
#                     </div>
#                 </div>
#                 <div class="band-stats">
#                     <div class="band-stat">
#                         <div class="metric-label">C-band</div>
#                         <div>Util: <span id="kspff-c-util">0</span>%</div>
#                         <div>Frag: <span id="kspff-c-frag">0</span></div>
#                     </div>
#                     <div class="band-stat">
#                         <div class="metric-label">L-band</div>
#                         <div>Util: <span id="kspff-l-util">0</span>%</div>
#                         <div>Frag: <span id="kspff-l-frag">0</span></div>
#                     </div>
#                 </div>
#             </div>
#         </div>
        
#         <div class="chart-container">
#             <canvas id="osnrChart"></canvas>
#         </div>
        
#         <div class="status">
#             <span id="status">Connecting to server...</span>
#         </div>
#     </div>

#     <script>
#         const socket = io();
        
#         // OSNR Chart
#         const ctx = document.getElementById('osnrChart').getContext('2d');
#         const osnrChart = new Chart(ctx, {
#             type: 'line',
#             data: {
#                 labels: [],
#                 datasets: [
#                     {
#                         label: 'H-DQN OSNR Margin',
#                         data: [],
#                         borderColor: '#2196F3',
#                         backgroundColor: 'rgba(33, 150, 243, 0.1)',
#                         tension: 0.4,
#                         borderWidth: 2,
#                     },
#                     {
#                         label: 'KSP-FF OSNR Margin',
#                         data: [],
#                         borderColor: '#f44336',
#                         backgroundColor: 'rgba(244, 67, 54, 0.1)',
#                         tension: 0.4,
#                         borderWidth: 2,
#                     }
#                 ]
#             },
#             options: {
#                 responsive: true,
#                 maintainAspectRatio: false,
#                 scales: {
#                     y: {
#                         beginAtZero: true,
#                         title: {
#                             display: true,
#                             text: 'Avg OSNR Margin (dB)',
#                             color: '#fff'
#                         },
#                         ticks: { color: '#888' },
#                         grid: { color: '#333' }
#                     },
#                     x: {
#                         title: {
#                             display: true,
#                             text: 'Step',
#                             color: '#fff'
#                         },
#                         ticks: { color: '#888', maxTicksLimit: 10 },
#                         grid: { color: '#333' }
#                     }
#                 },
#                 plugins: {
#                     legend: {
#                         labels: { color: '#fff' }
#                     },
#                     title: {
#                         display: true,
#                         text: 'Average OSNR Margin Comparison',
#                         color: '#4CAF50',
#                         font: { size: 16, weight: 'bold' }
#                     }
#                 }
#             }
#         });
        
#         socket.on('connect', () => {
#             document.getElementById('status').innerHTML = 
#                 '<span class="connected">● Connected - Receiving live updates</span>';
#         });
        
#         socket.on('disconnect', () => {
#             document.getElementById('status').innerHTML = '● Disconnected';
#         });
        
#         socket.on('training_update', (data) => {
#             // Update H-DQN
#             if (data['H-DQN']) {
#                 const hdqn = data['H-DQN'];
#                 document.getElementById('hdqn-step').textContent = hdqn.step;
#                 document.getElementById('hdqn-blocking').textContent = hdqn.blocking_rate + '%';
#                 document.getElementById('hdqn-load').textContent = hdqn.network_load + ' E';
#                 document.getElementById('hdqn-osnr').textContent = hdqn.avg_osnr_margin + ' dB';
                
#                 if (hdqn.band_stats && hdqn.band_stats.length >= 2) {
#                     document.getElementById('hdqn-c-util').textContent = hdqn.band_stats[0].utilization;
#                     document.getElementById('hdqn-c-frag').textContent = hdqn.band_stats[0].fragmentation;
#                     document.getElementById('hdqn-l-util').textContent = hdqn.band_stats[1].utilization;
#                     document.getElementById('hdqn-l-frag').textContent = hdqn.band_stats[1].fragmentation;
#                 }
                
#                 // Add to chart
#                 if (osnrChart.data.labels.length === 0 || 
#                     osnrChart.data.labels[osnrChart.data.labels.length - 1] !== hdqn.step) {
#                     osnrChart.data.labels.push(hdqn.step);
#                     osnrChart.data.datasets[0].data.push(hdqn.avg_osnr_margin);
                    
#                     // Keep last 100 points
#                     if (osnrChart.data.labels.length > 100) {
#                         osnrChart.data.labels.shift();
#                         osnrChart.data.datasets[0].data.shift();
#                         osnrChart.data.datasets[1].data.shift();
#                     }
#                     osnrChart.update('none');
#                 }
#             }
            
#             // Update KSP-FF
#             if (data['KSP-FF']) {
#                 const kspff = data['KSP-FF'];
#                 document.getElementById('kspff-step').textContent = kspff.step;
#                 document.getElementById('kspff-blocking').textContent = kspff.blocking_rate + '%';
#                 document.getElementById('kspff-load').textContent = kspff.network_load + ' E';
#                 document.getElementById('kspff-osnr').textContent = kspff.avg_osnr_margin + ' dB';
                
#                 if (kspff.band_stats && kspff.band_stats.length >= 2) {
#                     document.getElementById('kspff-c-util').textContent = kspff.band_stats[0].utilization;
#                     document.getElementById('kspff-c-frag').textContent = kspff.band_stats[0].fragmentation;
#                     document.getElementById('kspff-l-util').textContent = kspff.band_stats[1].utilization;
#                     document.getElementById('kspff-l-frag').textContent = kspff.band_stats[1].fragmentation;
#                 }
                
#                 // Add to chart (sync with H-DQN steps)
#                 if (data['H-DQN']) {
#                     const idx = osnrChart.data.labels.indexOf(kspff.step);
#                     if (idx >= 0) {
#                         osnrChart.data.datasets[1].data[idx] = kspff.avg_osnr_margin;
#                         osnrChart.update('none');
#                     }
#                 }
#             }
#         });
#     </script>
# </body>
# </html>
# """
TRAINING_VIZ_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Training Visualization</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            padding: 15px;
        }
        .container { max-width: 1900px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #4CAF50;
            font-size: 2em;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }
        .algorithm-panel {
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 10px;
            padding: 20px;
        }
        .algorithm-panel.hdqn { border-color: #2196F3; }
        .algorithm-panel.kspff { border-color: #f44336; }
        .panel-title {
            font-size: 1.5em;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .hdqn .panel-title { color: #2196F3; }
        .kspff .panel-title { color: #f44336; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 15px;
        }
        .metric {
            background: #0a0a0a;
            padding: 12px;
            border-radius: 5px;
            border: 1px solid #333;
        }
        .metric-label {
            color: #888;
            font-size: 0.85em;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.3em;
            font-weight: bold;
            color: #4CAF50;
        }
        .band-stats {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .band-stat {
            flex: 1;
            background: #0a0a0a;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #333;
        }
        .chart-container {
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 10px;
            padding: 20px;
            height: 320px;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }
        .summary-panel {
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .summary-title {
            color: #4CAF50;
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 15px;
        }
        .summary-item {
            background: #0a0a0a;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #333;
        }
        .summary-label {
            color: #888;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        .summary-value {
            font-size: 1.1em;
            margin: 5px 0;
        }
        .hdqn-color { color: #2196F3; }
        .kspff-color { color: #f44336; }
        .improvement {
            font-size: 0.9em;
            margin-top: 5px;
            padding: 5px;
            border-radius: 3px;
        }
        .improvement.positive { background: rgba(76, 175, 80, 0.2); color: #4CAF50; }
        .improvement.negative { background: rgba(244, 67, 54, 0.2); color: #f44336; }
        .status {
            text-align: center;
            padding: 10px;
            background: #1a1a1a;
            borderRadius: 5px;
            margin-top: 15px;
            color: #888;
        }
        .connected { color: #4CAF50; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Parallel Training: H-DQN vs KSP-FF - Live Metrics Dashboard</h1>
        
        <div class="grid">
            <div class="algorithm-panel hdqn">
                <div class="panel-title">Hierarchical DQN</div>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-label">Episode</div>
                        <div class="metric-value" id="hdqn-episode">0</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Blocking Rate</div>
                        <div class="metric-value" id="hdqn-blocking">0%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Network Load</div>
                        <div class="metric-value" id="hdqn-load">0 E</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg OSNR Margin</div>
                        <div class="metric-value" id="hdqn-osnr">0.0 dB</div>
                    </div>
                </div>
                <div class="band-stats">
                    <div class="band-stat">
                        <div class="metric-label">C-band</div>
                        <div>Util: <span id="hdqn-c-util">0</span>%</div>
                        <div>Frag: <span id="hdqn-c-frag">0</span></div>
                    </div>
                    <div class="band-stat">
                        <div class="metric-label">L-band</div>
                        <div>Util: <span id="hdqn-l-util">0</span>%</div>
                        <div>Frag: <span id="hdqn-l-frag">0</span></div>
                    </div>
                </div>
            </div>
            
            <div class="algorithm-panel kspff">
                <div class="panel-title">KSP-FF Baseline</div>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-label">Episode</div>
                        <div class="metric-value" id="kspff-episode">0</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Blocking Rate</div>
                        <div class="metric-value" id="kspff-blocking">0%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Network Load</div>
                        <div class="metric-value" id="kspff-load">0 E</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg OSNR Margin</div>
                        <div class="metric-value" id="kspff-osnr">0.0 dB</div>
                    </div>
                </div>
                <div class="band-stats">
                    <div class="band-stat">
                        <div class="metric-label">C-band</div>
                        <div>Util: <span id="kspff-c-util">0</span>%</div>
                        <div>Frag: <span id="kspff-c-frag">0</span></div>
                    </div>
                    <div class="band-stat">
                        <div class="metric-label">L-band</div>
                        <div>Util: <span id="kspff-l-util">0</span>%</div>
                        <div>Frag: <span id="kspff-l-frag">0</span></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container"><canvas id="blockingChart"></canvas></div>
            <div class="chart-container"><canvas id="bitBlockingChart"></canvas></div>
            <div class="chart-container"><canvas id="fragChart"></canvas></div>
            <div class="chart-container"><canvas id="bitrateChart"></canvas></div>
            <div class="chart-container"><canvas id="maskedChart"></canvas></div>
            <div class="chart-container"><canvas id="osnrChart"></canvas></div>
        </div>
        
        <div class="summary-panel">
            <div class="summary-title">Performance Comparison (Last 100 Episodes)</div>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-label">Blocking Rate</div>
                    <div class="summary-value hdqn-color">H-DQN: <span id="sum-hdqn-blocking">0.000</span></div>
                    <div class="summary-value kspff-color">KSP-FF: <span id="sum-kspff-blocking">0.000</span></div>
                    <div class="improvement" id="imp-blocking">-</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Fragmentation</div>
                    <div class="summary-value hdqn-color">H-DQN: <span id="sum-hdqn-frag">0.00</span></div>
                    <div class="summary-value kspff-color">KSP-FF: <span id="sum-kspff-frag">0.00</span></div>
                    <div class="improvement" id="imp-frag">-</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Bitrate Served (Gbps)</div>
                    <div class="summary-value hdqn-color">H-DQN: <span id="sum-hdqn-bitrate">0.0</span></div>
                    <div class="summary-value kspff-color">KSP-FF: <span id="sum-kspff-bitrate">0.0</span></div>
                    <div class="improvement" id="imp-bitrate">-</div>
                </div>
            </div>
        </div>
        
        <div class="status"><span id="status">Connecting to server...</span></div>
    </div>

    <script>
        const socket = io();
        
        const chartConfig = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#fff', font: { size: 11 } } },
                title: { color: '#4CAF50', font: { size: 14, weight: 'bold' } }
            },
            scales: {
                y: { ticks: { color: '#888' }, grid: { color: '#333' } },
                x: { ticks: { color: '#888', maxTicksLimit: 10 }, grid: { color: '#333' } }
            }
        };
        
        const charts = {
            blocking: new Chart(document.getElementById('blockingChart'), {
                type: 'line',
                data: { labels: [], datasets: [
                    { label: 'H-DQN', data: [], borderColor: '#2196F3', backgroundColor: 'rgba(33,150,243,0.1)', tension: 0.4, borderWidth: 2 },
                    { label: 'KSP-FF', data: [], borderColor: '#f44336', backgroundColor: 'rgba(244,67,54,0.1)', tension: 0.4, borderWidth: 2 }
                ]},
                options: { ...chartConfig, plugins: { ...chartConfig.plugins, title: { ...chartConfig.plugins.title, display: true, text: 'Service Blocking Rate (%)' } } }
            }),
            bitBlocking: new Chart(document.getElementById('bitBlockingChart'), {
                type: 'line',
                data: { labels: [], datasets: [
                    { label: 'H-DQN', data: [], borderColor: '#2196F3', backgroundColor: 'rgba(33,150,243,0.1)', tension: 0.4, borderWidth: 2 },
                    { label: 'KSP-FF', data: [], borderColor: '#f44336', backgroundColor: 'rgba(244,67,54,0.1)', tension: 0.4, borderWidth: 2 }
                ]},
                options: { ...chartConfig, plugins: { ...chartConfig.plugins, title: { ...chartConfig.plugins.title, display: true, text: 'Bit Rate Blocking (%)' } } }
            }),
            frag: new Chart(document.getElementById('fragChart'), {
                type: 'line',
                data: { labels: [], datasets: [
                    { label: 'H-DQN', data: [], borderColor: '#2196F3', backgroundColor: 'rgba(33,150,243,0.1)', tension: 0.4, borderWidth: 2 },
                    { label: 'KSP-FF', data: [], borderColor: '#f44336', backgroundColor: 'rgba(244,67,54,0.1)', tension: 0.4, borderWidth: 2 }
                ]},
                options: { ...chartConfig, plugins: { ...chartConfig.plugins, title: { ...chartConfig.plugins.title, display: true, text: 'Network Fragmentation (RMS)' } } }
            }),
            bitrate: new Chart(document.getElementById('bitrateChart'), {
                type: 'line',
                data: { labels: [], datasets: [
                    { label: 'H-DQN', data: [], borderColor: '#2196F3', backgroundColor: 'rgba(33,150,243,0.1)', tension: 0.4, borderWidth: 2 },
                    { label: 'KSP-FF', data: [], borderColor: '#f44336', backgroundColor: 'rgba(244,67,54,0.1)', tension: 0.4, borderWidth: 2 }
                ]},
                options: { ...chartConfig, plugins: { ...chartConfig.plugins, title: { ...chartConfig.plugins.title, display: true, text: 'Bit Rate Served (Gbps)' } } }
            }),
            masked: new Chart(document.getElementById('maskedChart'), {
                type: 'line',
                data: { labels: [], datasets: [
                    { label: 'H-DQN Masked Invalid', data: [], borderColor: '#2196F3', backgroundColor: 'rgba(33,150,243,0.1)', tension: 0.4, borderWidth: 2 }
                ]},
                options: { ...chartConfig, plugins: { ...chartConfig.plugins, title: { ...chartConfig.plugins.title, display: true, text: 'Masked Invalid Blocks (H-DQN)' } } }
            }),
            osnr: new Chart(document.getElementById('osnrChart'), {
                type: 'line',
                data: { labels: [], datasets: [
                    { label: 'H-DQN', data: [], borderColor: '#2196F3', backgroundColor: 'rgba(33,150,243,0.1)', tension: 0.4, borderWidth: 2 },
                    { label: 'KSP-FF', data: [], borderColor: '#f44336', backgroundColor: 'rgba(244,67,54,0.1)', tension: 0.4, borderWidth: 2 }
                ]},
                options: { ...chartConfig, plugins: { ...chartConfig.plugins, title: { ...chartConfig.plugins.title, display: true, text: 'Average OSNR Margin (dB)' } } }
            })
        };
        
        const dataHistory = {
            'H-DQN': { blocking: [], bitBlocking: [], frag: [], bitrate: [], masked: [], osnr: [] },
            'KSP-FF': { blocking: [], bitBlocking: [], frag: [], bitrate: [], masked: [], osnr: [] }
        };
        
        socket.on('connect', () => {
            document.getElementById('status').innerHTML = '<span class="connected">● Connected - Receiving live updates</span>';
        });
        
        socket.on('disconnect', () => {
            document.getElementById('status').innerHTML = '● Disconnected';
        });
        
        socket.on('metrics_update', (data) => {
            const algo = data.algorithm;
            const episode = data.episode;
            
            // Update status panels
            if (algo === 'H-DQN') {
                document.getElementById('hdqn-episode').textContent = episode;
                document.getElementById('hdqn-blocking').textContent = data.blocking_rate.toFixed(2) + '%';
                document.getElementById('hdqn-load').textContent = data.network_load.toFixed(1) + ' E';
                document.getElementById('hdqn-osnr').textContent = data.avg_osnr_margin.toFixed(2) + ' dB';
                
                if (data.band_stats && data.band_stats.length >= 2) {
                    document.getElementById('hdqn-c-util').textContent = data.band_stats[0].utilization;
                    document.getElementById('hdqn-c-frag').textContent = data.band_stats[0].fragmentation;
                    document.getElementById('hdqn-l-util').textContent = data.band_stats[1].utilization;
                    document.getElementById('hdqn-l-frag').textContent = data.band_stats[1].fragmentation;
                }
            } else {
                document.getElementById('kspff-episode').textContent = episode;
                document.getElementById('kspff-blocking').textContent = data.blocking_rate.toFixed(2) + '%';
                document.getElementById('kspff-load').textContent = data.network_load.toFixed(1) + ' E';
                document.getElementById('kspff-osnr').textContent = data.avg_osnr_margin.toFixed(2) + ' dB';
                
                if (data.band_stats && data.band_stats.length >= 2) {
                    document.getElementById('kspff-c-util').textContent = data.band_stats[0].utilization;
                    document.getElementById('kspff-c-frag').textContent = data.band_stats[0].fragmentation;
                    document.getElementById('kspff-l-util').textContent = data.band_stats[1].utilization;
                    document.getElementById('kspff-l-frag').textContent = data.band_stats[1].fragmentation;
                }
            }
            
            // Store data
            dataHistory[algo].blocking.push(data.blocking_rate);
            dataHistory[algo].bitBlocking.push(data.bit_rate_blocking);
            dataHistory[algo].frag.push(data.network_frag);
            dataHistory[algo].bitrate.push(data.bitrate_served);
            dataHistory[algo].masked.push(data.masked_invalids);
            dataHistory[algo].osnr.push(data.avg_osnr_margin);
            
            // Update charts
            updateCharts(episode);
            updateSummary();
        });
        
        function updateCharts(episode) {
            const maxPoints = 100;
            
            if (dataHistory['H-DQN'].blocking.length > 0 && dataHistory['KSP-FF'].blocking.length > 0) {
                const idx = Math.min(dataHistory['H-DQN'].blocking.length, dataHistory['KSP-FF'].blocking.length) - 1;
                
                if (charts.blocking.data.labels.length === 0 || 
                    charts.blocking.data.labels[charts.blocking.data.labels.length - 1] !== episode) {
                    
                    Object.keys(charts).forEach(key => {
                        charts[key].data.labels.push(episode);
                        
                        if (key === 'masked') {
                            charts[key].data.datasets[0].data.push(dataHistory['H-DQN'][key][idx]);
                        } else {
                            charts[key].data.datasets[0].data.push(dataHistory['H-DQN'][key][idx]);
                            charts[key].data.datasets[1].data.push(dataHistory['KSP-FF'][key][idx]);
                        }
                        
                        if (charts[key].data.labels.length > maxPoints) {
                            charts[key].data.labels.shift();
                            charts[key].data.datasets.forEach(ds => ds.data.shift());
                        }
                        
                        charts[key].update('none');
                    });
                }
            }
        }
        
        function updateSummary() {
            const window = 100;
            
            if (dataHistory['H-DQN'].blocking.length >= 10 && dataHistory['KSP-FF'].blocking.length >= 10) {
                const hdqnBlocking = avg(dataHistory['H-DQN'].blocking.slice(-window));
                const kspffBlocking = avg(dataHistory['KSP-FF'].blocking.slice(-window));
                document.getElementById('sum-hdqn-blocking').textContent = hdqnBlocking.toFixed(4);
                document.getElementById('sum-kspff-blocking').textContent = kspffBlocking.toFixed(4);
                const impBlocking = ((kspffBlocking - hdqnBlocking) / kspffBlocking * 100);
                updateImprovement('imp-blocking', impBlocking, true);
                
                const hdqnFrag = avg(dataHistory['H-DQN'].frag.slice(-window));
                const kspffFrag = avg(dataHistory['KSP-FF'].frag.slice(-window));
                document.getElementById('sum-hdqn-frag').textContent = hdqnFrag.toFixed(2);
                document.getElementById('sum-kspff-frag').textContent = kspffFrag.toFixed(2);
                const impFrag = ((kspffFrag - hdqnFrag) / kspffFrag * 100);
                updateImprovement('imp-frag', impFrag, true);
                
                const hdqnBitrate = avg(dataHistory['H-DQN'].bitrate.slice(-window));
                const kspffBitrate = avg(dataHistory['KSP-FF'].bitrate.slice(-window));
                document.getElementById('sum-hdqn-bitrate').textContent = hdqnBitrate.toFixed(1);
                document.getElementById('sum-kspff-bitrate').textContent = kspffBitrate.toFixed(1);
                const impBitrate = ((hdqnBitrate - kspffBitrate) / kspffBitrate * 100);
                updateImprovement('imp-bitrate', impBitrate, false);
            }
        }
        
        function updateImprovement(id, value, lowerIsBetter) {
            const elem = document.getElementById(id);
            const isPositive = lowerIsBetter ? value > 0 : value > 0;
            elem.className = 'improvement ' + (isPositive ? 'positive' : 'negative');
            elem.textContent = (value > 0 ? '+' : '') + value.toFixed(2) + '% improvement';
        }
        
        function avg(arr) {
            return arr.reduce((a, b) => a + b, 0) / arr.length;
        }
    </script>
</body>
</html>
"""


# =============================================================================
# Modified Main
# =============================================================================

def main():
    print("\n" + "="*80)
    print("Parallel Training with Live Browser Visualization")
    print("="*80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 80)
    topology_path = '../topologies/indian_net_5-paths_new.h5'
    
    if not os.path.exists(topology_path):
        print(f"ERROR: Topology not found: {topology_path}")
        return
    
    with open(topology_path, 'rb') as f:
        topology = pickle.load(f)
    
    env_args = dict(
        topology=topology,
        seed=10,
        allow_rejection=False,
        j=5,
        mean_service_holding_time=200,
        mean_service_inter_arrival_time=1.0,
        episode_length=100,
        num_bands=2,
        bit_rates=[100, 200, 400]
    )
    
    NUM_EPISODES = 5000
    
    print(f"\nConfiguration:")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Browser Visualization: http://localhost:5000")
    print(f"  Matplotlib Plots: ENABLED")
    print("="*80)
    
    base_log_dir = './logs/hierarchical_parallel/'
    dqn_log_dir = os.path.join(base_log_dir, 'hdqn/')
    kspff_log_dir = os.path.join(base_log_dir, 'kspff/')
    os.makedirs(dqn_log_dir, exist_ok=True)
    os.makedirs(kspff_log_dir, exist_ok=True)
    
    metrics_queue = Queue(maxsize=1000)
    viz_queue = Queue(maxsize=500)
    stop_event = Event()
    
    print("\nStarting processes...\n")
    
    # Browser visualization
    browser_viz_process = Process(target=run_browser_visualization,
                                   args=(viz_queue, stop_event, 5000))
    browser_viz_process.start()
    
    # # Matplotlib plots
    # plot_process = Process(target=plot_parallel_metrics, 
    #                       args=(metrics_queue, stop_event, base_log_dir))
    # plot_process.start()
    
    time.sleep(3)  # Let servers start
    
    # Training processes
    dqn_process = Process(target=train_hierarchical_dqn_with_viz,
                         args=(env_args, dqn_log_dir, metrics_queue, viz_queue,
                               stop_event, device, NUM_EPISODES))
    dqn_process.start()
    
    kspff_process = Process(target=run_kspff_parallel_with_viz,
                           args=(env_args, kspff_log_dir, metrics_queue, viz_queue,
                                 stop_event, NUM_EPISODES))
    kspff_process.start()
    
    try:
        dqn_process.join()
        kspff_process.join()
        time.sleep(5)
        stop_event.set()
        plot_process.join(timeout=10)
        browser_viz_process.terminate()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Cleaning up...")
        stop_event.set()
        dqn_process.terminate()
        kspff_process.terminate()
        plot_process.terminate()
        browser_viz_process.terminate()
    
    print("\n" + "="*80)
    print("All processes complete!")
    print("="*80)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()