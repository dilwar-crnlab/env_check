"""
ksp_ff_agent.py

K-Shortest Path First Fit (KSP-FF) Heuristic Agent
Integrated with optical_rl_gym logging system.

Strategy:
1. Try paths in order (0 to k-1)
2. For each path, try bands in order (0 to num_bands-1)
3. For each band, use First Fit to find initial slot
4. Validate using environment's methods
5. Return first valid combination
6. If none valid, reject

Author: Optical RL Gym Team
"""

import copy
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
import os
import pickle
import networkx as nx


class KSP_FF_Agent:
    """
    K-Shortest Path First Fit (KSP-FF) heuristic agent.
    
    Features:
    - Systematic exploration of all path-band-slot combinations
    - First Fit spectrum allocation policy
    - Complete validation before action selection
    - Integration with ServiceRequestLogger for detailed tracking
    - Statistics collection for performance analysis
    """
    
    def __init__(self, env, logger=None, enable_detailed_logging=False):
        """
        Initialize KSP-FF agent.
        
        Args:
            env: RMSA environment instance
            logger: Optional ServiceRequestLogger for comprehensive logging
            enable_detailed_logging: Enable detailed decision logging (requires ServiceRequestLogger)
        """
        self.env = env
        self.name = "KSP-FF"
        
        # ====================================================================
        # LOGGING SETUP
        # ====================================================================
        if logger is None:
            # Create basic logger if none provided
            self.logger = logging.getLogger("ksp_ff_agent")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
                self.logger.addHandler(handler)
        else:
            self.logger = logger
        
        # Check if we have ServiceRequestLogger (comprehensive logging)
        self.has_service_logger = hasattr(logger, 'log_service_arrival')
        self.enable_detailed_logging = enable_detailed_logging and self.has_service_logger
        
        # Track last blocking reason
        self.last_blocking_reason = None
        
        # ====================================================================
        # STATISTICS
        # ====================================================================
        self.stats = {
            'total_requests': 0,
            'accepted': 0,
            'rejected': 0,
            'path_usage': [0] * env.k_paths,
            'band_usage': [0] * env.num_bands,
        }
        
        # Track blocking reasons encountered during decision process
        self.blocking_reasons_encountered = {
            'path_length_exceeded': 0,
            'no_modulation_available': 0,
            'spectrum_unavailable': 0,
            'osnr_threshold_violation': 0,
            'osnr_interference_violation': 0,
        }
        
        # Log agent initialization
        if self.has_service_logger and self.enable_detailed_logging:
            self.logger.service_logger.info(
                f"{self.name} Agent initialized with detailed logging enabled"
            )
    
    # ========================================================================
    # LOGGING METHODS
    # ========================================================================
    
    def _log_decision_start(self, service):
        """Log start of agent decision process"""
        if self.enable_detailed_logging:
            self.logger.service_logger.info(
                f"[Service {service.service_id}] {self.name}: Starting decision process"
            )
    
    def _log_trying_combination(self, service_id: int, path_idx: int, band: int, 
                                initial_slot: Optional[int], path_length: float):
        """Log agent trying a specific combination"""
        if self.enable_detailed_logging:
            band_name = "C-band" if band == 0 else "L-band"
            if initial_slot is not None:
                self.logger.service_logger.debug(
                    f"[Service {service_id}] {self.name}: Evaluating "
                    f"Path {path_idx} ({path_length:.1f}km), {band_name}, Slot {initial_slot}"
                )
            else:
                self.logger.service_logger.debug(
                    f"[Service {service_id}] {self.name}: Evaluating "
                    f"Path {path_idx} ({path_length:.1f}km), {band_name}"
                )
    
    def _log_validation_result(self, service_id: int, path_idx: int, band: int, 
                               initial_slot: Optional[int], is_valid: bool, 
                               blocking_reason: Optional[str] = None):
        """Log validation result for a combination"""
        if self.enable_detailed_logging:
            band_name = "C-band" if band == 0 else "L-band"
            if is_valid:
                self.logger.service_logger.info(
                    f"[Service {service_id}] {self.name}: ✓ Valid allocation found - "
                    f"Path {path_idx}, {band_name}, Slot {initial_slot}"
                )
            else:
                self.logger.validation_logger.debug(
                    f"[Service {service_id}] {self.name}: ✗ Invalid - "
                    f"Path {path_idx}, {band_name}" +
                    (f", Slot {initial_slot}" if initial_slot is not None else "") +
                    f" - Reason: {blocking_reason}"
                )
    
    def _log_final_decision(self, service_id: int, action: Tuple[int, int, int], 
                           accepted: bool):
        """Log final agent decision"""
        if self.enable_detailed_logging:
            path_idx, band, slot = action
            if accepted:
                band_name = "C-band" if band == 0 else "L-band"
                self.logger.service_logger.info(
                    f"[Service {service_id}] {self.name} DECISION: ACCEPT - "
                    f"Path {path_idx}, {band_name}, Slot {slot}"
                )
            else:
                self.logger.service_logger.info(
                    f"[Service {service_id}] {self.name} DECISION: REJECT - "
                    f"Reason: {self.last_blocking_reason}"
                )
    
    # ========================================================================
    # CORE AGENT METHODS
    # ========================================================================
    
    def find_first_fit_slot(self, path, band: int) -> Optional[int]:
        """
        Find first available spectrum slot using First Fit policy.
        
        Args:
            path: Path object
            band: Band index
            
        Returns:
            int: Initial slot index (local coordinates) or None if not found
        """
        # Get number of slots needed
        slots_needed = self.env.get_number_slots(
            path, self.env.num_bands, band, self.env.modulations
        )
        
        if slots_needed <= 0:
            return None
        
        # Get available slots array for this path-band
        available_slots = self.env.get_available_slots(path, band)
        
        # Get band capacity
        x, y = self.env.get_shift(band)
        band_capacity = y - x
        
        # First Fit search: find first contiguous block
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
    
    def validate_allocation(self, path, band: int, initial_slot: int) -> Tuple[bool, Optional[str]]:
        """
        Validate allocation using environment's methods WITHOUT calling step().
        Replicates validation logic from env.step() to pre-check validity.
        
        Args:
            path: Path object
            band: Band index
            initial_slot: Starting slot (local coordinates)
            
        Returns:
            tuple: (is_valid, blocking_reason)
                - is_valid: True if allocation is valid
                - blocking_reason: String describing why invalid, or None if valid
        """
        # Validation 1: Path length constraint
        max_reach = max(m['max_reach'] for m in self.env.modulations)
        if path.length > max_reach:
            return False, 'path_length_exceeded'
        
        # Validation 2: Modulation availability and get slots needed
        slots = self.env.get_number_slots(
            path, self.env.num_bands, band, self.env.modulations
        )
        if slots <= 0:
            return False, 'no_modulation_available'
        
        # Validation 3: Spectrum availability
        if not self.env.is_path_free(path, initial_slot, slots, band):
            return False, 'spectrum_unavailable'
        
        # Validation 4 & 5: OSNR validation
        # Convert to global coordinates
        x = self.env.get_shift(band)[0]
        initial_slot_global = initial_slot + x
        
        # Create temporary service for OSNR checks
        temp_service = copy.deepcopy(self.env.current_service)
        temp_service.bandwidth = slots * 12.5e9  # Hz
        temp_service.band = band
        temp_service.initial_slot = initial_slot_global
        temp_service.number_slots = slots
        temp_service.path = path
        temp_service.center_frequency = self.env._calculate_center_frequency(temp_service)
        
        # Get modulation format
        modulation = self.env.get_modulation_format(
            path, self.env.num_bands, band, self.env.modulations
        )
        temp_service.modulation_format = modulation['modulation']
        
        # Validation 4: OSNR threshold check
        osnr_db = self.env.osnr_calculator.calculate_osnr(temp_service, self.env.topology)
        if osnr_db < self.env.OSNR_th[temp_service.modulation_format]:
            return False, 'osnr_threshold_violation'
        
        # Validation 5: OSNR interference check on existing services
        if not self.env._check_existing_services_osnr(path, band, temp_service):
            return False, 'osnr_interference_violation'
        
        return True, None
    
    def select_action(self, observation: Any = None) -> Tuple[int, int, int]:
        """
        Select action using KSP-FF heuristic.
        
        Systematically tries all path-band combinations in order:
        - For each path (0 to k-1)
        - For each band (0 to num_bands-1)
        - Find First Fit slot
        - Validate allocation
        - Return first valid combination
        
        Args:
            observation: Environment observation (not used by this heuristic)
        
        Returns:
            tuple: (path_idx, band_idx, initial_slot) or rejection action
        """
        self.stats['total_requests'] += 1
        self.last_blocking_reason = None
        
        current_service = self.env.current_service
        
        # ====================================================================
        # LOGGING: Decision process start
        # ====================================================================
        self._log_decision_start(current_service)
        
        # Track all blocking reasons encountered during search
        blocking_reasons_encountered = []
        
        # Get k-shortest paths for current service
        k_paths = self.env.k_shortest_paths[
            current_service.source, 
            current_service.destination
        ]
        
        # ====================================================================
        # TRY ALL PATH-BAND COMBINATIONS IN ORDER
        # ====================================================================
        for path_idx, path in enumerate(k_paths):
            
            for band in range(self.env.num_bands):
                
                # ============================================================
                # PRE-VALIDATION: Check constraints before finding slot
                # ============================================================
                
                # Check 1: Path length constraint
                max_reach = max(m['max_reach'] for m in self.env.modulations)
                if path.length > max_reach:
                    self.blocking_reasons_encountered['path_length_exceeded'] += 1
                    blocking_reasons_encountered.append('path_length_exceeded')
                    
                    self._log_trying_combination(
                        current_service.service_id, path_idx, band, None, path.length
                    )
                    self._log_validation_result(
                        current_service.service_id, path_idx, band, None,
                        False, 'path_length_exceeded'
                    )
                    continue
                
                # Check 2: Modulation availability
                slots = self.env.get_number_slots(
                    path, self.env.num_bands, band, self.env.modulations
                )
                if slots <= 0:
                    self.blocking_reasons_encountered['no_modulation_available'] += 1
                    blocking_reasons_encountered.append('no_modulation_available')
                    
                    self._log_trying_combination(
                        current_service.service_id, path_idx, band, None, path.length
                    )
                    self._log_validation_result(
                        current_service.service_id, path_idx, band, None,
                        False, 'no_modulation_available'
                    )
                    continue
                
                # ============================================================
                # FIND FIRST FIT SLOT
                # ============================================================
                initial_slot = self.find_first_fit_slot(path, band)
                
                if initial_slot is None:
                    # No contiguous spectrum block found
                    self.blocking_reasons_encountered['spectrum_unavailable'] += 1
                    blocking_reasons_encountered.append('spectrum_unavailable')
                    
                    self._log_trying_combination(
                        current_service.service_id, path_idx, band, None, path.length
                    )
                    self._log_validation_result(
                        current_service.service_id, path_idx, band, None,
                        False, 'spectrum_unavailable'
                    )
                    continue
                
                # ============================================================
                # FULL VALIDATION (including OSNR)
                # ============================================================
                self._log_trying_combination(
                    current_service.service_id, path_idx, band, initial_slot, path.length
                )
                
                is_valid, blocking_reason = self.validate_allocation(
                    path, band, initial_slot
                )
                
                if is_valid:
                    # ========================================================
                    # VALID ALLOCATION FOUND - RETURN THIS ACTION
                    # ========================================================
                    action = (path_idx, band, initial_slot)
                    
                    self._log_validation_result(
                        current_service.service_id, path_idx, band, initial_slot,
                        True, None
                    )
                    self._log_final_decision(
                        current_service.service_id, action, True
                    )
                    
                    return action
                else:
                    # ========================================================
                    # VALIDATION FAILED - TRY NEXT COMBINATION
                    # ========================================================
                    if blocking_reason and blocking_reason in self.blocking_reasons_encountered:
                        self.blocking_reasons_encountered[blocking_reason] += 1
                    
                    if blocking_reason:
                        blocking_reasons_encountered.append(blocking_reason)
                    
                    self._log_validation_result(
                        current_service.service_id, path_idx, band, initial_slot,
                        False, blocking_reason
                    )
        
        # ====================================================================
        # NO VALID COMBINATION FOUND - REJECT
        # ====================================================================
        
        # Determine most relevant blocking reason
        if blocking_reasons_encountered:
            # Priority order (most specific to least specific)
            reason_priority = [
                'osnr_interference_violation',
                'osnr_threshold_violation',
                'spectrum_unavailable',
                'no_modulation_available',
                'path_length_exceeded'
            ]
            
            # Select highest priority reason that was encountered
            for priority_reason in reason_priority:
                if priority_reason in blocking_reasons_encountered:
                    self.last_blocking_reason = priority_reason
                    break
            
            # Fallback to first encountered
            if self.last_blocking_reason is None:
                self.last_blocking_reason = blocking_reasons_encountered[0]
        else:
            self.last_blocking_reason = "no_spectrum_available"
        
        # Store blocking reason in service object for environment tracking
        self.env.current_service.blocking_reason = self.last_blocking_reason
        
        # Return rejection action
        action = (self.env.k_paths, self.env.num_bands, self.env.num_spectrum_resources)
        
        self._log_final_decision(
            current_service.service_id, action, False
        )
        
        return action
    
    # ========================================================================
    # STATISTICS METHODS
    # ========================================================================
    
    def update_stats(self, action: Tuple[int, int, int], reward: float, info: Dict):
        """
        Update agent statistics after environment step.
        
        Args:
            action: Taken action (path_idx, band_idx, slot)
            reward: Received reward
            info: Environment info dictionary
        """
        path_idx, band_idx, _ = action
        
        if reward > 0:
            # Service accepted
            self.stats['accepted'] += 1
            if path_idx < self.env.k_paths:
                self.stats['path_usage'][path_idx] += 1
            if band_idx < self.env.num_bands:
                self.stats['band_usage'][band_idx] += 1
        else:
            # Service rejected
            self.stats['rejected'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive agent statistics.
        
        Returns:
            dict: Statistics including acceptance rates, path/band usage, etc.
        """
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
            'blocking_reasons_encountered': self.blocking_reasons_encountered.copy()
        }
    
    def reset_statistics(self):
        """Reset all agent statistics."""
        self.stats = {
            'total_requests': 0,
            'accepted': 0,
            'rejected': 0,
            'path_usage': [0] * self.env.k_paths,
            'band_usage': [0] * self.env.num_bands,
        }
        
        self.blocking_reasons_encountered = {
            'path_length_exceeded': 0,
            'no_modulation_available': 0,
            'spectrum_unavailable': 0,
            'osnr_threshold_violation': 0,
            'osnr_interference_violation': 0,
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def run_ksp_ff_simulation(env, agent, num_episodes: int = 10) -> Dict[str, Any]:
    """
    Run KSP-FF agent simulation.
    
    Args:
        env: RMSA environment instance
        agent: KSP_FF_Agent instance
        num_episodes: Number of episodes to run
        
    Returns:
        dict: Simulation results including episode statistics
    """
    episode_results = []
    
    for episode in range(num_episodes):
        obs = env.reset(only_episode_counters=(episode > 0))
        done = False
        episode_reward = 0
        
        while not done:
            # Agent selects action
            action = agent.select_action(obs)
            
            # Environment executes action
            obs, reward, done, info = env.step(action)
            
            # Update agent statistics
            agent.update_stats(action, reward, info)
            
            episode_reward += reward
        
        episode_results.append({
            'episode': episode,
            'total_reward': episode_reward,
            'blocking_rate': info['episode_service_blocking_rate'],
            'bit_rate_blocking_rate': info.get('episode_bit_rate_blocking_rate', 0),
        })
    
    # Get final agent statistics
    final_stats = agent.get_statistics()
    
    return {
        'episode_results': episode_results,
        'agent_statistics': final_stats
    }


# Alias for backward compatibility
KSP_FirstFit_Agent = KSP_FF_Agent


# ============================================================================
# MAIN (for standalone testing)
# ============================================================================

def main():
    """Main function for standalone testing with logging."""
    import sys
    
    # ========================================================================
    # 1. LOAD TOPOLOGY
    # ========================================================================
    topology_name = 'indian_net'
    k_paths = 5
    topology_path = f'../topologies/{topology_name}_{k_paths}-paths_new.h5'
    
    if not os.path.exists(topology_path):
        print(f"ERROR: Topology file not found: {topology_path}")
        sys.exit(1)
    
    with open(topology_path, 'rb') as f:
        topology = pickle.load(f)
    
    print(f"Loaded topology: {topology.number_of_nodes()} nodes, "
          f"{topology.number_of_edges()} edges")
    
    # ========================================================================
    # 2. SETUP LOGGING
    # ========================================================================
    from optical_rl_gym.logging_utils import ServiceRequestLogger, LoggingConfig
    
    log_config = LoggingConfig(
        detail_level='high',
        log_service_arrival=True,
        log_validation_steps=True,
        log_osnr_calculations=True,
        output_format='both',
        verbose=True
    )
    
    # ========================================================================
    # 3. CREATE ENVIRONMENT
    # ========================================================================
    from optical_rl_gym.envs import RMSAEnv
    
    env = RMSAEnv(
        topology=topology,
        seed=42,
        allow_rejection=False,
        mean_service_holding_time=50.0,
        mean_service_inter_arrival_time=0.1,
        episode_length=100,
        num_bands=2,
        bit_rates=[50, 200, 100],
        k_paths=k_paths,
        enable_logging=True,
        log_config=log_config,
        log_dir='logs/ksp_ff_test'
    )
    
    print("Environment created successfully")
    
    # ========================================================================
    # 4. CREATE AGENT
    # ========================================================================
    agent = KSP_FF_Agent(
        env=env,
        logger=env.request_logger,
        enable_detailed_logging=True
    )
    
    print(f"Agent created: {agent.name}")
    print(f"Logging enabled: {agent.enable_detailed_logging}")
    
    # ========================================================================
    # 5. RUN SIMULATION
    # ========================================================================
    print("\nRunning simulation...")
    results = run_ksp_ff_simulation(env, agent, num_episodes=1)
    
    # ========================================================================
    # 6. PRINT RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("SIMULATION RESULTS")
    print("="*70)
    
    for ep_result in results['episode_results']:
        print(f"Episode {ep_result['episode']}: "
              f"Reward={ep_result['total_reward']:.2f}, "
              f"Blocking={ep_result['blocking_rate']*100:.2f}%")
    
    print("\nAgent Statistics:")
    stats = results['agent_statistics']
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Acceptance Rate: {stats['acceptance_rate']*100:.2f}%")
    
    print("\nBlocking Reasons:")
    for reason, count in stats['blocking_reasons_encountered'].items():
        if count > 0:
            pct = (count / stats['total_requests']) * 100
            print(f"  {reason}: {count} ({pct:.2f}%)")
    
    # ========================================================================
    # 7. CLEANUP
    # ========================================================================
    if env.request_logger:
        env.request_logger.close()
        summary = env.request_logger.get_summary()
        print(f"\nLogs saved to: {summary['log_directory']}")
    
    print("\n" + "="*70)
    print("KSP-FF test completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
