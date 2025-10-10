import copy
import numpy as np
import logging
from typing import Tuple, Optional
import os
import pickle
import networkx as nx


class KSP_FF_Agent:
    """
    K-Shortest Path First Fit (KSP-FF) agent.
    Compatible with spectrum visualization.
    
    Simple strategy:
    1. Try paths in order (0 to k-1)
    2. For each path, try bands in order (0 to num_bands-1)
    3. For each band, use First Fit to find slot
    4. Validate using env's methods (without calling step)
    5. Return first valid combination
    6. If none valid, reject
    """
    
    def __init__(self, env, debug=False, logger=None):
        """
        Initialize KSP-FF agent.
        
        Args:
            env: RMSA environment instance
            debug: Enable detailed debug output
            logger: Optional logger instance
        """
        self.env = env
        self.debug = debug
        self.logger = logger or logging.getLogger("ksp_ff_agent")
        self.name = "KSP-FF"
        
        # Track last blocking reason for visualization
        self.last_blocking_reason = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'accepted': 0,
            'rejected': 0,
            'path_usage': [0] * env.k_paths,
            'band_usage': [0] * env.num_bands,
        }
        
        # Track ALL blocking reasons encountered (not just final reason)
        self.blocking_reasons_encountered = {
            'path_length_exceeded': 0,
            'no_modulation_available': 0,
            'spectrum_unavailable': 0,
            'osnr_threshold_violation': 0,
            'osnr_interference_violation': 0,
        }
    
    def find_first_fit_slot(self, path, band) -> Optional[int]:
        """
        Find first available spectrum slot using First Fit.
        
        Args:
            path: Path object
            band: Band index
            
        Returns:
            int: Initial slot index (local coordinates) or None
        """
        # Get slots needed
        slots_needed = self.env.get_number_slots(path, self.env.num_bands, band, self.env.modulations)
        
        if slots_needed <= 0:
            return None
        
        # Get available slots
        available_slots = self.env.get_available_slots(path, band)
        
        # Get band capacity
        x, y = self.env.get_shift(band)
        band_capacity = y - x
        
        # First Fit search
        consecutive_free = 0
        start_slot = None
        
        for slot_idx in range(band_capacity):
            if available_slots[slot_idx] == 1:
                if consecutive_free == 0:
                    start_slot = slot_idx
                consecutive_free += 1
                
                if consecutive_free >= slots_needed:
                    return start_slot
            else:
                consecutive_free = 0
                start_slot = None
        
        return None
    
    def validate_allocation(self, path, band, initial_slot) -> Tuple[bool, Optional[str]]:
        """
        Validate allocation using env's methods WITHOUT calling step().
        Replicates the validation logic from env.step().
        
        Args:
            path: Path object
            band: Band index
            initial_slot: Starting slot (local coordinates)
            
        Returns:
            tuple: (is_valid, blocking_reason)
        """
        # Check 1: Path length constraint
        max_reach = max(m['max_reach'] for m in self.env.modulations)
        if path.length > max_reach:
            return False, 'path_length_exceeded'
        
        # Check 2: Modulation availability and get slots needed
        slots = self.env.get_number_slots(path, self.env.num_bands, band, self.env.modulations)
        if slots <= 0:
            return False, 'no_modulation_available'
        
        # Check 3: Spectrum availability
        if not self.env.is_path_free(path, initial_slot, slots, band):
            return False, 'spectrum_unavailable'
        
        # Check 4 & 5: OSNR validation
        # Convert to global coordinates
        x = self.env.get_shift(band)[0]
        initial_slot_global = initial_slot + x
        
        # Create temporary service
        temp_service = copy.deepcopy(self.env.current_service)
        temp_service.bandwidth = slots * 12.5e9
        temp_service.band = band
        temp_service.initial_slot = initial_slot_global
        temp_service.number_slots = slots
        temp_service.path = path
        temp_service.center_frequency = self.env._calculate_center_frequency(temp_service)
        
        # Get modulation
        modulation = self.env.get_modulation_format(path, self.env.num_bands, band, self.env.modulations)
        temp_service.modulation_format = modulation['modulation']
        
        # Check OSNR threshold
        osnr_db = self.env.osnr_calculator.calculate_osnr(temp_service, self.env.topology)
        #print("OSNR_DB", osnr_db)
        if osnr_db < self.env.OSNR_th[temp_service.modulation_format]:
            return False, 'osnr_threshold_violation'
        
        # Check OSNR interference on existing services
        if not self.env._check_existing_services_osnr(path, band, temp_service):
            return False, 'osnr_interference_violation'
        
        return True, None
    
    def select_action(self, observation=None) -> Tuple[int, int, int]:
        """
        Select action by trying all combinations in order.
        
        Args:
            observation: Environment observation (not used by this heuristic)
        
        Order:
        - Path 0, Band 0, First Fit slot → validate
        - Path 0, Band 1, First Fit slot → validate
        - Path 1, Band 0, First Fit slot → validate
        - Path 1, Band 1, First Fit slot → validate
        - ...
        - Return first valid combination
        - If none valid, return rejection action
        
        Returns:
            tuple: (path_idx, band_idx, initial_slot) or rejection
        """
        self.stats['total_requests'] += 1
        self.last_blocking_reason = None  # Reset
        
        # Track encountered blocking reasons (for better diagnostics)
        blocking_reasons_encountered = []
        
        current_service = self.env.current_service
        k_paths = self.env.k_shortest_paths[current_service.source, current_service.destination]
        
        if self.debug:
            print(f"\n{'='*80}")
            print(f"Analyzing Service {current_service.service_id}")
            print(f"{'='*80}")
            print(f"Route: {current_service.source} -> {current_service.destination}")
            print(f"Bit rate: {current_service.bit_rate} Gbps")
            print(f"Available paths: {len(k_paths)}")
        
        # Try all combinations in order
        for path_idx, path in enumerate(k_paths):
            
            if self.debug:
                print(f"\nPath {path_idx}: length={path.length:.1f} km")
            
            for band in range(self.env.num_bands):
                
                band_name = "C-band" if band == 0 else "L-band"
                
                # Check 1: Path length constraint (BEFORE finding slots)
                max_reach = max(m['max_reach'] for m in self.env.modulations)
                if path.length > max_reach:
                    self.blocking_reasons_encountered['path_length_exceeded'] += 1
                    blocking_reasons_encountered.append('path_length_exceeded')
                    if self.debug:
                        print(f"  {band_name}: Path too long ({path.length:.1f} > {max_reach} km)")
                    continue
                
                # Check 2: Modulation availability (BEFORE finding slots)
                slots = self.env.get_number_slots(path, self.env.num_bands, band, self.env.modulations)
                if slots <= 0:
                    self.blocking_reasons_encountered['no_modulation_available'] += 1
                    blocking_reasons_encountered.append('no_modulation_available')
                    if self.debug:
                        print(f"  {band_name}: No modulation available")
                    continue
                
                # Check 3: Find First Fit slot
                initial_slot = self.find_first_fit_slot(path, band)
                
                if initial_slot is None:
                    # Track spectrum unavailability
                    self.blocking_reasons_encountered['spectrum_unavailable'] += 1
                    blocking_reasons_encountered.append('spectrum_unavailable')
                    
                    if self.debug:
                        print(f"  {band_name}: No contiguous spectrum block found")
                    continue
                
                # Get number of slots for debug output
                slots = self.env.get_number_slots(path, self.env.num_bands, band, self.env.modulations)
                
                if self.debug:
                    print(f"  {band_name}: Found slot {initial_slot}, needs {slots} slots")
                
                # Validate this combination
                is_valid, blocking_reason = self.validate_allocation(path, band, initial_slot)
                
                if is_valid:
                    # Found valid allocation
                    if self.debug:
                        print(f"  {band_name}: ✓ VALID - Accepting allocation")
                        print(f"\n{'='*80}")
                        print(f"Service {current_service.service_id} ACCEPTED")
                        print(f"{'='*80}\n")
                    
                    self.logger.info(
                        f"Service {current_service.service_id}: "
                        f"Accepted - Path {path_idx} ({path.length:.1f}km), "
                        f"Band {band}, Slot {initial_slot}"
                    )
                    return (path_idx, band, initial_slot)
                else:
                    # Increment counter for this specific blocking reason
                    if blocking_reason and blocking_reason in self.blocking_reasons_encountered:
                        self.blocking_reasons_encountered[blocking_reason] += 1
                    
                    # Track for priority selection
                    if blocking_reason:
                        blocking_reasons_encountered.append(blocking_reason)
                    
                    # This combination failed - try next
                    if self.debug:
                        print(f"  {band_name}: ✗ Failed - {blocking_reason}")
                    
                    self.logger.debug(
                        f"Service {current_service.service_id}: "
                        f"Path {path_idx}, Band {band}, Slot {initial_slot} - "
                        f"Failed: {blocking_reason}"
                    )
        
        # No valid combination found
        # Determine the most relevant blocking reason from all encountered
        if blocking_reasons_encountered:
            # Priority order (most specific to least specific)
            reason_priority = [
                'osnr_interference_violation',
                'osnr_threshold_violation',
                'spectrum_unavailable',
                'no_modulation_available',
                'path_length_exceeded'
            ]
            
            # Select the highest priority reason that was encountered
            for priority_reason in reason_priority:
                if priority_reason in blocking_reasons_encountered:
                    self.last_blocking_reason = priority_reason
                    break
            
            # If no priority match, use first encountered
            if self.last_blocking_reason is None:
                self.last_blocking_reason = blocking_reasons_encountered[0]
        else:
            self.last_blocking_reason = "no_spectrum_available"
        
        # CRITICAL: Store the blocking reason in the service object
        # so the environment can track it correctly
        self.env.current_service.blocking_reason = self.last_blocking_reason
        
        if self.debug:
            print(f"\n{'='*80}")
            print(f"Service {current_service.service_id} BLOCKED: {self.last_blocking_reason}")
            print(f"{'='*80}\n")
        
        self.logger.info(
            f"Service {current_service.service_id}: "
            f"Rejected - All combinations failed ({self.last_blocking_reason})"
        )
        
        return (self.env.k_paths, self.env.num_bands, self.env.num_spectrum_resources)
    
    def update_stats(self, action, reward, info):
        """Update statistics after environment step."""
        path_idx, band_idx, _ = action
        
        if reward > 0:
            self.stats['accepted'] += 1
            if path_idx < self.env.k_paths:
                self.stats['path_usage'][path_idx] += 1
            if band_idx < self.env.num_bands:
                self.stats['band_usage'][band_idx] += 1
        else:
            self.stats['rejected'] += 1
    
    def get_statistics(self) -> dict:
        """Get agent statistics."""
        total = self.stats['total_requests']
        
        if total == 0:
            return self.stats
        
        return {
            'total_requests': total,
            'accepted': self.stats['accepted'],
            'rejected': self.stats['rejected'],
            'acceptance_rate': self.stats['accepted'] / total,
            'blocking_rate': self.stats['rejected'] / total,
            'path_usage_distribution': [
                count / self.stats['accepted'] if self.stats['accepted'] > 0 else 0
                for count in self.stats['path_usage']
            ],
            'band_usage_distribution': [
                count / self.stats['accepted'] if self.stats['accepted'] > 0 else 0
                for count in self.stats['band_usage']
            ],
        }
    
    def reset_statistics(self):
        """Reset agent statistics."""
        self.stats = {
            'total_requests': 0,
            'accepted': 0,
            'rejected': 0,
            'path_usage': [0] * self.env.k_paths,
            'band_usage': [0] * self.env.num_bands,
        }


# Alias for backward compatibility with visualization script
KSP_FirstFit_Agent = KSP_FF_Agent


def run_ksp_ff_simulation(env, num_episodes=10, debug=False):
    """
    Run KSP-FF agent simulation.
    
    Args:
        env: RMSA environment instance
        num_episodes: Number of episodes to run
        debug: Enable debug output
        
    Returns:
        dict: Simulation results
    """
    agent = KSP_FF_Agent(env, debug=debug)
    
    episode_results = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Agent tries all combinations and returns first valid one
            action = agent.select_action(obs)
            
            # Execute in environment (should always match agent's validation)
            obs, reward, done, info = env.step(action)
            
            # Update statistics
            agent.update_stats(action, reward, info)
            
            episode_reward += reward
        
        episode_results.append({
            'episode': episode,
            'total_reward': episode_reward,
            'blocking_rate': info['episode_service_blocking_rate'],
            'bit_rate_blocking_rate': info['episode_bit_rate_blocking_rate'],
        })
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Blocking={info['episode_service_blocking_rate']:.4f}")
    
    # Final statistics
    final_stats = agent.get_statistics()
    
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Total Requests: {final_stats['total_requests']}")
    print(f"Accepted: {final_stats['accepted']}")
    print(f"Rejected: {final_stats['rejected']}")
    print(f"Acceptance Rate: {final_stats['acceptance_rate']:.4f}")
    print(f"Blocking Rate: {final_stats['blocking_rate']:.4f}")
    print(f"\nPath Usage: {final_stats['path_usage_distribution']}")
    print(f"Band Usage: {final_stats['band_usage_distribution']}")
    print("="*60)
    
    return {
        'episode_results': episode_results,
        'agent_statistics': final_stats
    }


# def main():
#     """Main function for standalone testing."""
#     # Configuration
#     topology_name = 'indian_net'
#     k_paths = 5
#     topology_path = f'../topologies/{topology_name}_{k_paths}-paths_new.h5'

#     print(f"\nLoading topology: {topology_path}")

#     if not os.path.exists(topology_path):
#         print(f"ERROR: Topology not found: {topology_path}")
#         print(f"Trying alternative path...")
        
#         # Try GraphML format
#         topology_path = f'./topologies/{topology_name}.graphml'
#         if os.path.exists(topology_path):
#             topology = nx.read_graphml(topology_path, node_type=int)
#             print(f"Loaded from GraphML: {topology_path}")
#         else:
#             print(f"ERROR: No topology file found")
#             return
#     else:
#         with open(topology_path, 'rb') as f:
#             topology = pickle.load(f)

#     print(f"Topology: {topology.number_of_nodes()} nodes, {topology.number_of_edges()} edges")

#     # Import environment
#     from optical_rl_gym.envs import RMSAEnv

#     # Environment configuration
#     print("\nCreating RMSAEnv...")
#     env = RMSAEnv(
#         topology=topology,
#         seed=42,
#         allow_rejection=False,
#         mean_service_holding_time=200.0,
#         mean_service_inter_arrival_time=1.0,
#         episode_length=100,
#         num_bands=2,
#         bit_rates=[50, 100, 200],
#         k_paths=k_paths,
#         load=100.0
#     )

#     print("Environment created successfully")

#     # Run simulation with debug enabled for first episode
#     print(f"\n{'='*80}")
#     print("KSP-FF Simulation Starting")
#     print(f"{'='*80}")
    
#     results = run_ksp_ff_simulation(env, num_episodes=1, debug=True)

#     print(f"\n{'='*80}")
#     print("KSP-FF evaluation finished successfully!")
#     print(f"{'='*80}")


# if __name__ == "__main__":
#     main()