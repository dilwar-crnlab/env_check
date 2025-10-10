


"""
run_spectrum_visualization.py

Simple script to visualize spectrum allocation/deallocation in real-time.
"""

import os
import pickle
import numpy as np
import networkx as nx
from spectrum_vis import SpectrumVisualizer


def run_visualization_demo():
    """Run KSP-FF with spectrum visualization."""
    
    print("="*80)
    print("Spectrum Visualization Demo")
    print("="*80)
    
    # Import environment and agent
    from optical_rl_gym.envs import RMSAEnv
    from ksp_ff_osnr_check import KSP_FF_Agent
    
    # Load topology
    topology_name = 'indian_net'
    k_paths = 5
    topology_path = f'../topologies/{topology_name}_{k_paths}-paths_new.h5'
    
    print(f"\nLoading topology: {topology_path}")
    
    if not os.path.exists(topology_path):
        print(f"ERROR: Topology not found: {topology_path}")
        print("Please check the path and ensure the file exists.")
        return
    
    with open(topology_path, 'rb') as f:
        topology = pickle.load(f)
    
    print(f"Loaded: {topology.number_of_nodes()} nodes, {topology.number_of_edges()} edges")
    
    # Create environment
    print("\nCreating RMSAEnv...")
    env = RMSAEnv(
        topology=topology,
        seed=42,
        episode_length=1500000,
        num_bands=2,
        bit_rates=[200, 400, 1000],
        k_paths=5,
        mean_service_holding_time=900.0,
        mean_service_inter_arrival_time=1.0
    )
    
    # Create agent with debug mode
    agent = KSP_FF_Agent(env, debug=True)  # Enable detailed debug output
    
    # Create visualizer
    save_dir = './spectrum_plots/'
    visualizer = SpectrumVisualizer(env, save_dir=save_dir, interactive=False)
    
    print(f"\nSaving plots to: {save_dir}")
    print("\nStarting visualization...")
    print("="*80)
    
    # Reset environment
    obs = env.reset()
    
    # Plot initial empty spectrum
    visualizer.plot_spectrum(title_suffix="- Initial Empty Spectrum")
    print(f"Step 0: Initial state plotted")
    
    # Run for N steps
    num_steps = 1500000
    done = False
    step = 0
    
    while step < num_steps and not done:
        # Get action from KSP-FF
        action = agent.select_action(obs)
        
        # Execute
        obs, reward, done, info = env.step(action)
        step += 1
        
        # Determine what happened
        #service = env.current_service
        service = env.topology.graph["services"][-1]
        if service.accepted:
            event = "Allocation"
            color = "✓"
        else:
            event = "Rejection"
            color = "✗"
            
        # Plot current spectrum state
        title = f"Service {service.service_id}"
        visualizer.plot_spectrum(
            title_suffix=title,
            event_type=event.lower(),
            service_id=service.service_id
        )
        
        # Print summary
        print(f"Step {step}: {color} {event} | "
              f"Service {service.service_id} | "
              f"Src: {service.source} -> Dst: {service.destination} | "
              f"Bit rate: {service.bit_rate} Gbps | "
              f"Reward: {reward:.2f}")
        
        if service.accepted:
            print(f"         Path: {action[0]}, Band: {action[1]}, "
                  f"Slot: {action[2]}, Slots: {service.number_slots}")
            if hasattr(service, 'modulation_format'):
                print(f"         Modulation: {service.modulation_format}")
            if hasattr(service, 'OSNR_margin'):
                print(f"         OSNR margin: {service.OSNR_margin:.2f} dB")
        else:
            # Get blocking reason from agent
            blocking_reason = agent.last_blocking_reason or "unknown"
            print(f"         BLOCKED: {blocking_reason}")
            
            # Show additional details from environment if available
            if hasattr(service, 'blocking_reason'):
                env_reason = service.blocking_reason
                if env_reason and env_reason != blocking_reason:
                    print(f"         (Env says: {env_reason})")
        
        print()  # Blank line between steps
        
        # Plot fragmentation every 10 steps
        if step % 10 == 0:
            visualizer.plot_fragmentation_analysis()
            print(f"         >>> Fragmentation analysis saved")
        
        print()
    
    visualizer.close()
    
    print("="*80)
    print(f"Visualization complete!")
    print(f"Total plots created: {visualizer.step_count}")
    print(f"Location: {save_dir}")
    print("="*80)
    
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"  Services processed: {env.services_processed}")
    print(f"  Services accepted:  {env.services_accepted}")
    print(f"  Blocking rate:      {(env.services_processed - env.services_accepted) / env.services_processed * 100:.2f}%")
    
    # Band utilization
    num_edges = env.topology.number_of_edges()
    available_slots = env.topology.graph.get('available_slots')
    
    if available_slots is not None:
        c_start, c_end = env.get_shift(0)
        c_capacity = (c_end - c_start) * num_edges
        c_free = np.sum(available_slots[0:num_edges, c_start:c_end])
        c_util = (c_capacity - c_free) / c_capacity * 100
        
        print(f"  C-band utilization: {c_util:.2f}%")
        
        if env.num_bands > 1:
            l_start, l_end = env.get_shift(1)
            l_capacity = (l_end - l_start) * num_edges
            l_free = np.sum(available_slots[num_edges:2*num_edges, l_start:l_end])
            l_util = (l_capacity - l_free) / l_capacity * 100
            print(f"  L-band utilization: {l_util:.2f}%")
    
    print("\nTo view plots, open the PNG files in:", save_dir)


if __name__ == "__main__":
    run_visualization_demo()