import copy
import functools
import heapq
import logging
import math
from collections import defaultdict
from typing import Optional, Sequence, Tuple

import gym
import networkx as nx
import numpy as np
import random
from optical_rl_gym.utils import Path, Service
from optical_rl_gym.osnr_calculator import *

from .optical_network_env import OpticalNetworkEnv


class RMSAEnv(OpticalNetworkEnv):

    metadata = {
        "metrics": [
            "service_blocking_rate",
            "episode_service_blocking_rate",
            "bit_rate_blocking_rate",
            "episode_bit_rate_blocking_rate",
        ]
    }

    # Band configuration constants - defined in single place
    BAND_CONFIG = {
        'C_BAND': {
            'id': 0,
            'slots': 344,
            'start_slot': 0,
            'end_slot': 343
        },
        'L_BAND': {
            'id': 1,
            'slots': 480,  # 824 - 344
            'start_slot': 344,
            'end_slot': 823
        }
    }
    
    # Total slots for different band configurations
    SPECTRUM_RESOURCES_CONFIG = {
        1: BAND_CONFIG['C_BAND']['slots'],  # C-band only: 344 slots
        2: BAND_CONFIG['L_BAND']['end_slot'] + 1  # C+L bands: 824 slots total
    }

    def __init__(
        self,
        num_bands=None,
        topology: nx.Graph = None,
        episode_length: int = 1000,
        mean_service_holding_time: float = 0.0,
        mean_service_inter_arrival_time: float =0.0,
        load: float =0,
        bit_rates = [50, 100, 200],
        bit_rate_probabilities: Optional[np.array] = None,
        node_request_probabilities: Optional[np.array] = None,
        seed: Optional[int] = None,
        allow_rejection: bool = False,
        reset: bool = True,
        channel_width: float = 12.5,
        k_paths=5,
        j=1
    ):
        super().__init__(
            topology,
            episode_length=episode_length,
            load=mean_service_holding_time/mean_service_inter_arrival_time,
            mean_service_holding_time=mean_service_holding_time,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            allow_rejection=allow_rejection,
            channel_width=channel_width,
            k_paths=k_paths
        )

        # make sure that modulations are set in the topology
        #assert "modulations" in self.topology.graph

        # ADD THIS: Store bit rates and probabilities
        self.bit_rates = list(bit_rates)
        

        self.j = j
        
        if bit_rate_probabilities is not None:
            assert len(bit_rate_probabilities) == len(self.bit_rates), \
                "bit_rate_probabilities length must match bit_rates length"
            self.bit_rate_probabilities = bit_rate_probabilities
        else:
            # Uniform probability if not specified
            self.bit_rate_probabilities = [1.0 / len(self.bit_rates)] * len(self.bit_rates)
    
        self.physical_params = PhysicalParameters() # for using PhysicalParameters data class
        # Initialize OSNR calculator
        self.osnr_calculator = OSNRCalculator()
        self.num_bands = num_bands
        # specific attributes for elastic optical networks

        self.bit_rate_requested = 0
        self.bit_rate_accepted = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0

        # Initialize network fragmentation tracking
        self.topology.graph["current_network_fragmentation"] = 0.0
        self.topology.graph["current_band_fragmentation"] = {band: 0.0 for band in range(self.num_bands)}
        self.topology.graph["previous_network_fragmentation"] = 0.0
        
        # Initialize OSNR tracking dictionaries
        self._affected_services_osnr_before = {}
        self._affected_services_osnr_after = {}


            # Curriculum Learning Parameters
        self.use_curriculum = True  # Enable/disable curriculum learning
        self.curriculum_stage_1_episodes = 3000
        self.curriculum_stage_2_episodes = 8000
        self.adaptive_curriculum = False  # Use performance-based progression
        
        # Reward function weights
        self.w_routing = 0.2
        self.w_band = 0.2
        self.w_spectrum = 0.15
        self.lambda_distance = 1.5
        
        # Service classification thresholds
        #self.ml_avg = 2  # Average modulation level (BPSK=1, QPSK=2, 8QAM=3, 16QAM=4)
        self.ml_avg = self._calculate_ml_avg()
        #self.f_avg = 4   # Average required slots (calculated above)
        self.f_avg = self._calculate_f_avg()
        
        # Episode tracking for curriculum
        self.total_episodes = 0
        self.current_curriculum_stage = 1
        
        # For adaptive curriculum
        self.recent_routing_bonuses = []
        self.recent_band_accuracies = []
        self.routing_mastery_threshold = 0.12  # 60% of max routing bonus
        self.band_accuracy_threshold = 0.80
        
        # Component tracking for analysis
        self.reward_components_history = []



        num_edges = self.topology.number_of_edges()
        
        # Use centralized band configuration
        self.num_spectrum_resources = self.SPECTRUM_RESOURCES_CONFIG[self.num_bands]

        #bit error rate (BER) of 10−3 are 9 dB, 12dB, 16 dB, and 18.6 dB,
        self.OSNR_th ={
            'PM_BPSK': 9,
            'PM_QPSK': 12,
            'PM_8QAM': 16,
            'PM_16QAM': 18.6
        }

        # Frequency ranges for C, L bands (in Hz)
        self.band_frequencies = {
            self.BAND_CONFIG['C_BAND']['id']: {  # C-band
                'start': 191.7e12,  # Hz
                'end': 196.0e12,    # Hz
            },
            self.BAND_CONFIG['L_BAND']['id']: {  # L-band
                'start': 185.7e12,  # Hz (corrected from 184.4e12)
                'end': 191.7e12,    # Hz (corrected from 191.3e12)
            }
        }

        self.spectrum_usage = np.zeros(
            (self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources), dtype=int
        )
        # matrix to store the spectrum allocation
        self.spectrum_slots_allocation = np.full(
            (self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources),
            fill_value=-1,
            dtype=int,
        )

        # ADD THIS: Initialize blocking reason tracking
        self.blocking_reasons = {
            'invalid_action': 0,
            'path_length_exceeded': 0,
            'no_modulation_available': 0,
            'spectrum_unavailable': 0,
            'osnr_threshold_violation': 0,
            'osnr_interference_violation': 0,
            'total_blocked': 0,
            'total_accepted': 0
        }
        
        self.episode_blocking_reasons = {
            'invalid_action': 0,
            'path_length_exceeded': 0,
            'no_modulation_available': 0,
            'spectrum_unavailable': 0,
            'osnr_threshold_violation': 0,
            'osnr_interference_violation': 0,
            'total_blocked': 0,
            'total_accepted': 0
        }

        # do we allow proactive rejection or not?
        self.reject_action = 1 if allow_rejection else 0

        # defining the observation and action spaces
        self.actions_output = np.zeros((self.k_paths + 1, 
                                        self.num_bands + 1,
                                        self.num_spectrum_resources + 1), dtype=int
        )
        self.episode_actions_output = np.zeros(
            (self.k_paths + 1, self.num_bands + 1, self.num_spectrum_resources + 1), dtype=int
        )
        self.actions_taken = np.zeros((self.k_paths + 1, self.num_bands + 1, self.num_spectrum_resources + 1), dtype=int)
        
        self.episode_actions_taken = np.zeros((self.k_paths + 1, self.num_bands + 1, self.num_spectrum_resources + 1), dtype=int)
        
        self.action_space = gym.spaces.MultiDiscrete(
            (
                self.k_paths + self.reject_action,
                self.num_bands + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
            )
        )
        self.observation_space = gym.spaces.Dict(
            {
                "topology": gym.spaces.Discrete(10),
                "current_service": gym.spaces.Discrete(10),
            }
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger("rmsaenv")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                "Logging is enabled for DEBUG which generates a large number of messages. "
                "Set it to INFO if DEBUG is not necessary."
            )
        self._new_service = False
        if reset:
            self.reset(only_episode_counters=False)

    def _calculate_center_frequency(self, service: Service) -> float:
        """
        Calculate the center frequency of a service allocation.
        
        Note: service.initial_slot is stored in GLOBAL coordinates (0-823)
        after being shifted by _provision_path().
        
        Args:
            service: Service object with allocation details
            
        Returns:
            float: Center frequency in Hz
            
        Example:
            C-band allocation at global slots 100-109:
                - Band start: 191.7 THz
                - Local center: slot 104.5
                - Frequency: 191.7 THz + (104.5 × 12.5 GHz) = 193.006 THz
                
            L-band allocation at global slots 400-409:
                - Band start: 185.7 THz  
                - Local slots: 56-65 (400-344 to 409-344)
                - Local center: slot 60.5
                - Frequency: 185.7 THz + (60.5 × 12.5 GHz) = 186.456 THz
        """
        # Step 1: Calculate center slot in GLOBAL coordinates
        global_start_slot = service.initial_slot
        global_end_slot = service.initial_slot + service.number_slots - 1
        global_center_slot = (global_start_slot + global_end_slot) / 2.0
        
        # Step 2: Convert to BAND-LOCAL coordinates
        # Get the starting slot index for this band
        band_start_slot = self.get_shift(service.band)[0]
        local_center_slot = global_center_slot - band_start_slot
        
        # Step 3: Calculate center frequency
        # Start from band's base frequency and add offset
        band_base_frequency = self.band_frequencies[service.band]['start']
        frequency_offset = local_center_slot * self.channel_width * 1e9  # Convert GHz to Hz
        center_freq = band_base_frequency + frequency_offset
        
        # Optional: Validate frequency is within band limits
        band_end_frequency = self.band_frequencies[service.band]['end']
        if not (band_base_frequency <= center_freq <= band_end_frequency):
            self.logger.warning(
                f"Service {service.service_id} center frequency {center_freq/1e12:.3f} THz "
                f"outside band {service.band} range "
                f"[{band_base_frequency/1e12:.3f}, {band_end_frequency/1e12:.3f}] THz"
            )
        
        return center_freq

    # Add this method to calculate average OSNR margins
    def _calculate_avg_osnr_margin(self):
        """
        Calculate the average OSNR margin across all active connections.
        
        Returns:
            float: Average OSNR margin in dB, or 0 if no connections are active
        """
        running_services = self.topology.graph["running_services"]
        
        if not running_services:
            return 0.0
        
        # Calculate current OSNR for all running services
        total_margin = 0.0
        services_with_margin = 0
        
        for service in running_services:
            # Some older services might not have OSNR margin tracked
            if hasattr(service, 'OSNR_margin'):
                total_margin += service.OSNR_margin
                services_with_margin += 1
            else:
                # Recalculate OSNR for services that don't have it stored
                osnr_db = self.osnr_calculator.calculate_osnr(service, self.topology)
                if hasattr(service, 'modulation_format') and service.modulation_format in self.OSNR_th:
                    osnr_threshold = self.OSNR_th[service.modulation_format]
                    margin = osnr_db - osnr_threshold
                    # Store it for future reference
                    service.current_OSNR = osnr_db
                    service.OSNR_th = osnr_threshold
                    service.OSNR_margin = margin
                    
                    total_margin += margin
                    services_with_margin += 1
        
        # Return average margin or 0 if no services had margin data
        return total_margin / services_with_margin if services_with_margin > 0 else 0.0

    # Add this to track OSNR margins per band
    def _calculate_band_osnr_margins(self):
        """
        Calculate the average OSNR margin for each band.
        
        Returns:
            dict: Dictionary with average OSNR margin per band
        """
        running_services = self.topology.graph["running_services"]
        
        if not running_services:
            return {band: 0.0 for band in range(self.num_bands)}
        
        # Initialize counters and sum for each band
        band_margins = {band: 0.0 for band in range(self.num_bands)}
        band_counts = {band: 0 for band in range(self.num_bands)}
        
        for service in running_services:
            if hasattr(service, 'band') and hasattr(service, 'OSNR_margin'):
                band = service.band
                band_margins[band] += service.OSNR_margin
                band_counts[band] += 1
        
        # Calculate averages
        avg_band_margins = {
            band: band_margins[band] / band_counts[band] if band_counts[band] > 0 else 0.0
            for band in range(self.num_bands)
        }
        
        return avg_band_margins
    
    def step(self, action: [int]):
        path, band, initial_slot = action[0], action[1], action[2]
        self.actions_output[path, band, initial_slot] += 1
        
        # Track bit rate for current service
        current_bit_rate = self.current_service.bit_rate
        self.bit_rate_requested += current_bit_rate
        self.episode_bit_rate_requested += current_bit_rate
        
        # Start with service rejected
        self.current_service.accepted = False
        blocking_reason = None
        
        # Check if action is valid (route/band/spectrum availability check)
        if not (path < self.k_paths and band < self.num_bands and initial_slot < self.num_spectrum_resources):
            blocking_reason = 'invalid_action'
        else:
            temp_path = self.k_shortest_paths[self.current_service.source, self.current_service.destination][path]
            
            # Check path length constraint
            max_reach = max(m['max_reach'] for m in self.modulations)
            if temp_path.length > max_reach:
                blocking_reason = 'path_length_exceeded'
            else:
                # Calculate slots needed (modulation check)
                slots = self.get_number_slots(temp_path, self.num_bands, band, self.modulations)
                
                if slots <= 0:
                    blocking_reason = 'no_modulation_available'
                else:
                    # Check if path is free (spectrum availability)
                    if not self.is_path_free(temp_path, initial_slot, slots, band):
                        blocking_reason = 'spectrum_unavailable'
                    else:
                        # FIX: Shift initial_slot to global coordinates
                        x = self.get_shift(band)[0]
                        initial_slot_shift = initial_slot + x
                        # Create temporary service for OSNR checks
                        temp_service = copy.deepcopy(self.current_service)
                        temp_service.bandwidth = slots * 12.5e9  # in Hz
                        temp_service.band = band
                        temp_service.initial_slot = initial_slot_shift  
                        temp_service.number_slots = slots
                        temp_service.path = temp_path
                        temp_service.center_frequency = self._calculate_center_frequency(temp_service)
                        
                        # Get modulation format
                        modulation = self.get_modulation_format(temp_path, self.num_bands, band, self.modulations)['modulation']
                        temp_service.modulation_format = modulation
                        
                        # First check OSNR for the new service (SNR condition check)
                        osnr_db = self.osnr_calculator.calculate_osnr(temp_service, self.topology)
                        
                        if osnr_db < self.OSNR_th[temp_service.modulation_format]:
                            blocking_reason = 'osnr_threshold_violation'
                        else:
                            # Check if adding this service would affect existing services' OSNR
                            # This method also stores OSNR before/after values for reward calculation
                            if not self._check_existing_services_osnr(temp_path, band, temp_service):
                                blocking_reason = 'osnr_interference_violation'
                            else:
                                # Service can be accepted - all conditions satisfied
                                self.current_service.current_OSNR = osnr_db
                                self.current_service.OSNR_th = self.OSNR_th[temp_service.modulation_format]
                                self.current_service.OSNR_margin = self.current_service.current_OSNR - self.current_service.OSNR_th
                                
                                # Provision the path (this will update network fragmentation)
                                self._provision_path(temp_path, initial_slot, slots, band, self.current_service.arrival_time)
                                self.current_service.accepted = True
                                self.current_service.modulation_format = modulation
                                self.actions_taken[path, band, initial_slot] += 1
                                self._add_release(self.current_service)
                                
                                # Update acceptance counters
                                self.blocking_reasons['total_accepted'] += 1
                                self.episode_blocking_reasons['total_accepted'] += 1
                                
                                # Update bit rate acceptance counters
                                self.bit_rate_accepted += current_bit_rate
                                self.episode_bit_rate_accepted += current_bit_rate
        
        # Record blocking reason if service was not accepted
        if not self.current_service.accepted:
            self.actions_taken[self.k_paths, self.num_bands, self.num_spectrum_resources] += 1
            
            # Clear OSNR tracking if service was blocked
            if hasattr(self, '_affected_services_osnr_before'):
                self._affected_services_osnr_before.clear()
            if hasattr(self, '_affected_services_osnr_after'):
                self._affected_services_osnr_after.clear()
            
            # Update blocking reason counters
            if blocking_reason:
                self.blocking_reasons[blocking_reason] += 1
                self.episode_blocking_reasons[blocking_reason] += 1
                self.blocking_reasons['total_blocked'] += 1
                self.episode_blocking_reasons['total_blocked'] += 1
                
                # Store blocking reason in current service for analysis
                self.current_service.blocking_reason = blocking_reason
                
                # Log detailed blocking information
                self.logger.debug(f"Service {self.current_service.service_id} blocked due to: {blocking_reason}")
        
        # Add service to history
        self.topology.graph["services"].append(self.current_service)
        
        # Get path for reward calculation
        k_paths = self.k_shortest_paths[self.current_service.source, self.current_service.destination]
        path_selected = k_paths[path] if path < self.k_paths else None
        reward = self.reward(band, path_selected)
        
        # Clean up OSNR tracking after reward calculation
        if hasattr(self, '_affected_services_osnr_before'):
            self._affected_services_osnr_before.clear()
        if hasattr(self, '_affected_services_osnr_after'):
            self._affected_services_osnr_after.clear()

        # Calculate blocking rates by reason
        total_processed = self.services_processed
        episode_total_processed = self.episode_services_processed
        
        blocking_rate_breakdown = {}
        episode_blocking_rate_breakdown = {}
        
        if total_processed > 0:
            for reason in self.blocking_reasons:
                if reason not in ['total_blocked', 'total_accepted']:
                    blocking_rate_breakdown[f"{reason}_rate"] = self.blocking_reasons[reason] / total_processed
        
        if episode_total_processed > 0:
            for reason in self.episode_blocking_reasons:
                if reason not in ['total_blocked', 'total_accepted']:
                    episode_blocking_rate_breakdown[f"episode_{reason}_rate"] = self.episode_blocking_reasons[reason] / episode_total_processed

        # Calculate bit rate blocking rates
        bit_rate_blocking_rate = (self.bit_rate_requested - self.bit_rate_accepted) / self.bit_rate_requested if self.bit_rate_requested > 0 else 0.0
        episode_bit_rate_blocking_rate = (self.episode_bit_rate_requested - self.episode_bit_rate_accepted) / self.episode_bit_rate_requested if self.episode_bit_rate_requested > 0 else 0.0

        # Create info dictionary with metrics including blocking reasons and fragmentation
        info = {
            "band": band if self.services_accepted else -1,
            "service_blocking_rate": (self.services_processed - self.services_accepted) / self.services_processed,
            "episode_service_blocking_rate": (self.episode_services_processed - self.episode_services_accepted) / self.episode_services_processed,
            "bit_rate_blocking_rate": bit_rate_blocking_rate,
            "episode_bit_rate_blocking_rate": episode_bit_rate_blocking_rate,
            "blocking_reason": blocking_reason if blocking_reason else "accepted",
            "blocking_reasons_count": self.blocking_reasons.copy(),
            "episode_blocking_reasons_count": self.episode_blocking_reasons.copy(),
            "network_fragmentation": self.topology.graph.get("current_network_fragmentation", 0.0),
            "band_fragmentation": self.topology.graph.get("current_band_fragmentation", {}).copy(),
            **blocking_rate_breakdown,
            **episode_blocking_rate_breakdown
        }

        self._new_service = False
        self._next_service()
        return (self.observation(), reward, self.episode_services_processed == self.episode_length, info,)

 
    def _calculate_adaptive_weights(self):
        """
        Calculate adaptive beta and gamma weights based on network utilization.
        
        Returns:
            tuple: (beta, gamma) - fragmentation and OSNR weights
        """
        # Calculate network-wide utilization
        total_slots = self.num_spectrum_resources * self.topology.number_of_edges() * self.num_bands
        occupied_slots = total_slots - np.sum(self.topology.graph["available_slots"])
        utilization = occupied_slots / total_slots if total_slots > 0 else 0.0
        
        # Store for monitoring
        self.current_utilization = utilization
        
        # Adaptive weight schedule
        if utilization < 0.3:
            # Low utilization: Learn acceptance, light quality focus
            beta, gamma = 0.2, 0.15
        elif utilization < 0.5:
            # Medium-low: Transition to quality awareness
            progress = (utilization - 0.3) / 0.2
            beta = 0.2 + 0.3 * progress   # 0.2 → 0.5
            gamma = 0.15 + 0.25 * progress  # 0.15 → 0.4
        elif utilization < 0.7:
            # Medium-high: Quality becomes important
            progress = (utilization - 0.5) / 0.2
            beta = 0.5 + 0.15 * progress   # 0.5 → 0.65
            gamma = 0.4 + 0.15 * progress  # 0.4 → 0.55
        else:
            # High utilization: Maximum quality focus
            progress = min(1.0, (utilization - 0.7) / 0.2)
            beta = 0.65 + 0.15 * progress   # 0.65 → 0.8
            gamma = 0.55 + 0.2 * progress   # 0.55 → 0.75
        
        return beta, gamma


    def reset(self, only_episode_counters=True):
        """
        Reset environment for new episode.
        
        Args:
            only_episode_counters: If True, soft reset (between episodes)
                                If False, hard reset (full reinitialization)
        """
        # ========================================================================
        # EPISODE-LEVEL RESET (happens every episode)
        # ========================================================================
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        
        # Reset episode bit rate tracking
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_accepted = 0
        
        # Reset episode action tracking
        self.episode_actions_output = np.zeros(
            (
                self.k_paths + self.reject_action,
                self.num_bands + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
            ),
            dtype=int,
        )
        self.episode_actions_taken = np.zeros(
            (
                self.k_paths + self.reject_action,
                self.num_bands + self.reject_action,
                self.num_spectrum_resources + self.reject_action,
            ),
            dtype=int,
        )




        
        # Reset episode blocking reasons
        if hasattr(self, 'episode_blocking_reasons'):
            self.episode_blocking_reasons = {
                'invalid_action': 0,
                'path_length_exceeded': 0,
                'no_modulation_available': 0,
                'spectrum_unavailable': 0,
                'osnr_threshold_violation': 0,
                'osnr_interference_violation': 0,
                'total_blocked': 0,
                'total_accepted': 0
            }
        
        # ========================================================================
        # CURRICULUM LEARNING: EPISODE TRACKING
        # ========================================================================
        if only_episode_counters:
            # Increment total episodes for curriculum progression
            self.total_episodes += 1
            
            # Check for curriculum stage transitions
            if self.use_curriculum:
                old_stage = self._get_curriculum_stage()
                
                # For fixed curriculum, check if we've crossed thresholds
                if not self.adaptive_curriculum:
                    if (old_stage == 1 and 
                        self.total_episodes == self.curriculum_stage_1_episodes):
                        print(f"\n{'='*70}")
                        print(f"{'CURRICULUM STAGE 1 → 2':^70}")
                        print(f"{'='*70}")
                        print(f"Episode {self.total_episodes}: Adding Band Selection Component")
                        print(f"Reward = base + routing + BAND")
                        print(f"{'='*70}\n")
                    
                    elif (old_stage == 2 and 
                        self.total_episodes == self.curriculum_stage_2_episodes):
                        print(f"\n{'='*70}")
                        print(f"{'CURRICULUM STAGE 2 → 3':^70}")
                        print(f"{'='*70}")
                        print(f"Episode {self.total_episodes}: Adding Spectrum Allocation Component")
                        print(f"Reward = base + routing + band + SPECTRUM")
                        print(f"{'='*70}\n")
                
                # Print curriculum statistics every 1000 episodes
                if self.total_episodes % 1000 == 0:
                    stats = self.get_curriculum_statistics()
                    if 'error' not in stats:
                        print(f"\n{'='*70}")
                        print(f"Episode {stats['total_episodes']:,} | "
                            f"Stage {stats['current_stage']}/3 | "
                            f"Accept: {stats['acceptance_rate']*100:.1f}%")
                        print(f"{'-'*70}")
                        print(f"  Routing Mastery:     {stats['routing_mastery_%']:5.1f}% "
                            f"(avg bonus: {stats['avg_routing_bonus']:.3f})")
                        print(f"  Band Accuracy:       {stats['band_accuracy']*100:5.1f}% "
                            f"(correct assignments)")
                        print(f"  Spectrum Quality:    {stats['avg_spectrum_bonus']:.3f} "
                            f"(FF/LF policy adherence)")
                        print(f"  Average Total Reward: {stats['avg_total_reward']:.3f}")
                        print(f"{'='*70}\n")
            
            # Increment service counter if there's a pending service
            if self._new_service:
                self.episode_services_processed += 1
            
            return self.observation()
        
        # ========================================================================
        # FULL RESET (happens at training start or explicit reset)
        # ========================================================================
        super().reset()
        
        # Reset cumulative bit rate tracking
        self.bit_rate_requested = 0
        self.bit_rate_accepted = 0
        
        # Reset cumulative blocking reasons
        if hasattr(self, 'blocking_reasons'):
            self.blocking_reasons = {
                'invalid_action': 0,
                'path_length_exceeded': 0,
                'no_modulation_available': 0,
                'spectrum_unavailable': 0,
                'osnr_threshold_violation': 0,
                'osnr_interference_violation': 0,
                'total_blocked': 0,
                'total_accepted': 0
            }
        
        # Reset fragmentation tracking
        self.topology.graph["current_network_fragmentation"] = 0.0
        self.topology.graph["current_band_fragmentation"] = {
            band: 0.0 for band in range(self.num_bands)
        }
        self.topology.graph["previous_network_fragmentation"] = 0.0
        
        # Reset OSNR tracking
        self._affected_services_osnr_before = {}
        self._affected_services_osnr_after = {}
        
        # Reset spectrum allocation
        num_edges = self.topology.number_of_edges()
        self.topology.graph["available_slots"] = np.ones(
            (num_edges * self.num_bands, self.num_spectrum_resources), 
            dtype=int
        )
        self.spectrum_slots_allocation = np.full(
            (num_edges * self.num_bands, self.num_spectrum_resources),
            fill_value=-1, 
            dtype=int
        )
        
        # ========================================================================
        # CURRICULUM LEARNING: FULL RESET
        # ========================================================================
        # Reset curriculum tracking (only on full reset)
        if not hasattr(self, 'total_episodes'):
            self.total_episodes = 0
        
        if not hasattr(self, 'current_curriculum_stage'):
            self.current_curriculum_stage = 1
        
        if not hasattr(self, 'reward_components_history'):
            self.reward_components_history = []
        
        if not hasattr(self, 'recent_routing_bonuses'):
            self.recent_routing_bonuses = []
        
        if not hasattr(self, 'recent_band_accuracies'):
            self.recent_band_accuracies = []
        
        # Print initialization message
        if self.use_curriculum:
            print(f"\n{'='*70}")
            print(f"{'CURRICULUM LEARNING INITIALIZED':^70}")
            print(f"{'='*70}")
            print(f"Mode: {'Adaptive (performance-based)' if self.adaptive_curriculum else 'Fixed episodes'}")
            if not self.adaptive_curriculum:
                print(f"  Stage 1 (Routing):          Episodes 0 - {self.curriculum_stage_1_episodes:,}")
                print(f"  Stage 2 (+ Band):           Episodes {self.curriculum_stage_1_episodes:,} - {self.curriculum_stage_2_episodes:,}")
                print(f"  Stage 3 (+ Spectrum):       Episodes {self.curriculum_stage_2_episodes:,}+")
            else:
                print(f"  Stage 1 → 2 trigger:        Routing mastery ≥ {self.routing_mastery_threshold:.2f}")
                print(f"  Stage 2 → 3 trigger:        Band accuracy ≥ {self.band_accuracy_threshold:.0%}")
            print(f"\nReward Weights:")
            print(f"  Routing bonus (w_r):        {self.w_routing}")
            print(f"  Band bonus (w_b):           {self.w_band}")
            print(f"  Spectrum bonus (w_s):       {self.w_spectrum}")
            print(f"\nClassification Thresholds:")
            print(f"  ML_avg:                     {self.ml_avg} (BPSK/QPSK → L, 8QAM/16QAM → C)")
            print(f"  F_avg:                      {self.f_avg} slots (FF/LF policy threshold)")
            print(f"{'='*70}\n")
        
        # Generate first service
        self._new_service = False
        self._next_service()
        
        return self.observation()

    def render(self, mode="human"):
        return
    
    def _check_existing_services_osnr(self, path: Path, band: int, new_service: Service) -> bool:
        """
        Check if adding the new service would affect the OSNR of existing services on the same path.
        Also stores OSNR before/after values for reward calculation.
        
        Args:
            path: The path for the new service
            band: The band for the new service
            new_service: The new service to be added
            
        Returns:
            bool: True if all existing services maintain acceptable OSNR, False otherwise
        """
        # Get all services that share at least one link with the new service path
        affected_services = []
        
        # Extract the links from the new service path
        new_path_links = [(path.node_list[i], path.node_list[i+1]) 
                        for i in range(len(path.node_list)-1)]
        
        # Get all running services from the topology
        running_services = self.topology.graph["running_services"]
        
        # Initialize storage for OSNR tracking
        self._affected_services_osnr_before = {}
        self._affected_services_osnr_after = {}
        
        # For each running service, check if it shares links with the new service
        for service in running_services:
            service_path_links = [(service.path.node_list[i], service.path.node_list[i+1]) 
                                for i in range(len(service.path.node_list)-1)]
            
            # Check for common links (intersection)
            common_links = set(new_path_links).intersection(set(service_path_links))
            
            if common_links:
                # Store OSNR before adding new service
                if hasattr(service, 'current_OSNR'):
                    self._affected_services_osnr_before[service.service_id] = service.current_OSNR
                
                affected_services.append(service)
        
        # If no services are affected, return True
        if not affected_services:
            return True
        
        # Temporarily add the new service to the topology to simulate its effect
        original_services = self.topology.graph["running_services"].copy()
        self.topology.graph["running_services"].append(new_service)
        
        # Check OSNR for all affected services
        all_osnr_ok = True
        
        for service in affected_services:
            # Calculate the new OSNR with the added service
            osnr_db = self.osnr_calculator.calculate_osnr(service, self.topology)
            
            # Store OSNR after adding new service
            self._affected_services_osnr_after[service.service_id] = osnr_db
            
            # Check if OSNR is still above threshold
            if osnr_db < self.OSNR_th[service.modulation_format]:
                all_osnr_ok = False
                break

        # If all services maintain acceptable OSNR, update their OSNR values
        if all_osnr_ok:
            for service in affected_services:
                osnr_db = self._affected_services_osnr_after[service.service_id]
                service.current_OSNR = osnr_db
                service.OSNR_margin = osnr_db - self.OSNR_th[service.modulation_format]
                self.logger.debug(f"Updated Service {service.service_id} OSNR: {osnr_db} dB, " 
                                f"Margin: {service.OSNR_margin} dB")
        else:
            # Clear the after values if check failed (service will be rejected)
            self._affected_services_osnr_after.clear()
        
        # Restore the original services list
        self.topology.graph["running_services"] = original_services
        
        return all_osnr_ok

    def _next_service(self):
        if self._new_service:
            return
    
        at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        self.current_time = at

        ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        src, src_id, dst, dst_id = self._get_node_pair()

        # generate the bit rate according to the selection adopted
        #bit_rate = self.rng.choices(self.bit_rates, weights=self.bit_rate_probabilities)[0]
        bit_rate = self.rng.choices(self.bit_rates)[0]


        self.current_service = Service(
            self.episode_services_processed,
            src,
            src_id,
            destination=dst,
            destination_id=dst_id,
            arrival_time=at,
            holding_time=ht,
            bit_rate=bit_rate,
        )
        self._new_service = True

        self.services_processed += 1
        self.episode_services_processed += 1

        # release connections up to this point
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:  # release is not to be processed yet
                self._add_release(service_to_release)  # puts service back in the queue
                break  # breaks the loop


    def get_shift(self, band):
        """Get slot range for a specific band using centralized configuration"""
        if band == self.BAND_CONFIG['C_BAND']['id']:
            return (self.BAND_CONFIG['C_BAND']['start_slot'], 
                    self.BAND_CONFIG['C_BAND']['end_slot'] + 1)
        elif band == self.BAND_CONFIG['L_BAND']['id']:
            return (self.BAND_CONFIG['L_BAND']['start_slot'], 
                    self.BAND_CONFIG['L_BAND']['end_slot'] + 1)
        else:
            raise ValueError(f"Invalid band ID: {band}")
    
    def is_path_free(self, path: Path, initial_slot: int, number_slots: int, band) -> bool:
        x = self.get_shift(band)[0]
        initial_slot_shift = initial_slot + x
        y = self.get_shift(band)[1]  # End of band
        if initial_slot_shift + number_slots > y:
            return False
        for i in range(len(path.node_list) - 1):
            if np.any(self.topology.graph["available_slots"][
                    ((self.topology[path.node_list[i]][path.node_list[i + 1]]["index"]) +
                    (self.topology.number_of_edges() * band)),
                    initial_slot_shift : initial_slot_shift + number_slots] == 0):
                return False
        return True

    def get_available_slots(self, path: Path, band):
        """Modified to handle directed links"""
        x = self.get_shift(band)[0]
        y = self.get_shift(band)[1]
        
        # For directed path, only consider links in the path direction
        available_slots = functools.reduce(
            np.multiply,
            self.topology.graph["available_slots"][[((self.topology[path.node_list[i]][path.node_list[i + 1]]['index']) +
                (self.topology.number_of_edges() * band))
                for i in range(len(path.node_list) - 1)], x:y])
        
        return available_slots

    def rle(inarray):
        """run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values)"""
        # from: https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
        ia = np.asarray(inarray)  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)  # must include last element posi
            z = np.diff(np.append(-1, i))  # run lengths
            p = np.cumsum(np.append(0, z))[:-1]  # positions
            return p, ia[i], z

    def get_available_blocks(self, path, num_bands, band, modulations):
        # get available slots across the whole path
        # 1 if slot is available across all the links
        # zero if not
        available_slots = self.get_available_slots(
            self.k_shortest_paths[
                self.current_service.source, self.current_service.destination
            ][path], band
        )

        # getting the number of slots necessary for this service across this path
        slots = self.get_number_slots(
            self.k_shortest_paths[
                self.current_service.source, self.current_service.destination
            ][path], num_bands, band, modulations
        )

        # getting the blocks
        initial_indices, values, lengths = RMSAEnv.rle(available_slots)

        # selecting the indices where the block is available, i.e., equals to one
        available_indices = np.where(values == 1)

        # selecting the indices where the block has sufficient slots
        sufficient_indices = np.where(lengths >= slots)

        # getting the intersection, i.e., indices where the slots are available in sufficient quantity
        # and using only the J first indices
        final_indices = np.intersect1d(available_indices, sufficient_indices)[: self.j]

        return initial_indices[final_indices], lengths[final_indices]

    def get_number_slots(self, path: Path, num_bands, band, modulations) -> int:
        """
        Method that computes the number of spectrum slots necessary to accommodate the service request into the path.
        The method already adds the guardband.
        """
        max_reach = max(modulation['max_reach'] for modulation in modulations)
        if path.length > max_reach:
            return -1  # Path length exceeds max reach of all modulation formats
        modulation = self.get_modulation_format(path, num_bands, band, modulations)
        #print("Modulation", modulation)
        service_bit_rate = self.current_service.bit_rate
        number_of_slots = math.ceil(service_bit_rate / modulation['capacity']) + 1
        return number_of_slots
    
    def calculate_MF(self, modulations, length):
        for i in range(len(modulations) - 1):
            if length > modulations[i + 1]['max_reach']:
                if length <= modulations[i]['max_reach']:
                    return modulations[i]
        return modulations[len(modulations) - 1]
    
    def get_modulation_format(self, path: Path, num_bands, band, modulations):
        length = path.length
        if num_bands == 1:  # C band only
            modulation_format = self.calculate_MF(modulations, length)
        elif num_bands == 2:  # C + L band
            if band == self.BAND_CONFIG['C_BAND']['id']:  # C band
                modulation_format = self.calculate_MF(modulations, length)
            elif band == self.BAND_CONFIG['L_BAND']['id']:  # L band
                modulation_format = self.calculate_MF(modulations, length)

        return modulation_format 

    # Modulation format
    #Ref. Paper "ISRS impact-reduced routing, modulation, band, and spectrum allocation algorithm in 
    # C + L-bands elastic optical networks"
    # [BPSK, QPSK, 8QAM, 16QAM]
    capacity = [12.5, 25, 37.5, 50]
    #capacity = [50, 100, 150, 200]
    modulations = list()
    modulations.append({'modulation': 'PM_BPSK', 'capacity': capacity[0], 'max_reach': 4000})
    modulations.append({'modulation': 'PM_QPSK', 'capacity': capacity[1], 'max_reach': 2000})
    modulations.append({'modulation': 'PM_8QAM', 'capacity': capacity[2], 'max_reach': 1000})
    modulations.append({'modulation': 'PM_16QAM', 'capacity': capacity[3], 'max_reach': 500})

    def calculate_link_entropy_fragmentation(self, link_idx: int, band: int):
        """
        Shannon entropy-based fragmentation metric.
        Normalized to [0, 1] range.
        
        Returns:
            0 = No fragmentation (one contiguous block)
            1 = Maximum fragmentation (uniformly distributed tiny blocks)
        """
        x, y = self.get_shift(band)
        num_edges = self.topology.number_of_edges()
        offset = link_idx + (num_edges * band)
        
        spectrum = self.topology.graph["available_slots"][offset, x:y]
        
        # Find free blocks
        initial_indices, values, lengths = RMSAEnv.rle(spectrum)
        available_indices = np.where(values == 1)[0]
        
        if len(available_indices) == 0:
            return 1.0  # No free slots = max fragmentation
        
        block_sizes = lengths[available_indices]
        total_free = np.sum(block_sizes)
        
        if total_free == 0:
            return 1.0
        
        if len(block_sizes) == 1:
            return 0.0  # One contiguous block = no fragmentation
        
        # Calculate probabilities
        probabilities = block_sizes / total_free
        
        # Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Normalize by max possible (each slot separate)
        max_entropy = np.log2(total_free) if total_free > 1 else 1.0
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy

    def calculate_network_fragmentation(self):
        """
        Network fragmentation = average entropy across all link-bands.
        
        Simple, interpretable, treats all links equally.
        """
        num_links = self.topology.number_of_edges()
        total_entropy = 0.0
        valid_count = 0
        
        # Per-band tracking
        band_entropy_totals = {band: 0.0 for band in range(self.num_bands)}
        band_counts = {band: 0 for band in range(self.num_bands)}
        
        for link_idx in range(num_links):
            for band in range(self.num_bands):
                entropy = self.calculate_link_entropy_fragmentation(link_idx, band)
                
                total_entropy += entropy
                valid_count += 1
                
                band_entropy_totals[band] += entropy
                band_counts[band] += 1
        
        # Network average
        network_fragmentation = total_entropy / valid_count if valid_count > 0 else 0.0
        
        # Per-band average
        band_fragmentation = {
            band: band_entropy_totals[band] / band_counts[band]
            if band_counts[band] > 0 else 0.0
            for band in range(self.num_bands)
        }
        
        return {
            "network_fragmentation": network_fragmentation,
            "band_fragmentation": band_fragmentation,
        }


    def _update_network_fragmentation(self):
        """
        Update network fragmentation metrics.
        Stores previous value before updating for reward calculation.
        Called after each allocation and deallocation.
        """
        # Store previous value before updating
        self.topology.graph["previous_network_fragmentation"] = self.topology.graph.get(
            "current_network_fragmentation", 0.0
        )
        
        # Calculate new fragmentation
        fragmentation_metrics = self.calculate_network_fragmentation()
        
        # Store current fragmentation for monitoring
        self.current_network_fragmentation = fragmentation_metrics["network_fragmentation"]
        self.current_band_fragmentation = fragmentation_metrics["band_fragmentation"]
        
        # Store in topology graph for access
        self.topology.graph["current_network_fragmentation"] = self.current_network_fragmentation
        self.topology.graph["current_band_fragmentation"] = self.current_band_fragmentation

    def _provision_path(self, path: Path, initial_slot, number_slots, band, at):
        """Modified to validate allocation integrity"""
        
        # Computing horizontal shift using centralized band configuration
        x = self.get_shift(band)[0]
        initial_slot_shift = initial_slot + x
        num_edges = self.topology.number_of_edges()
        band_offset_multiplier = num_edges * band

        # Cache service for efficiency
        current_service = self.current_service
        available_slots = self.topology.graph["available_slots"]
        path_nodes = path.node_list
        
        # For directed path, we only update the links in the direction of the path
        for i in range(len(path_nodes) - 1):
            node1, node2 = path_nodes[i], path_nodes[i + 1]
            
            # Validate edge exists
            if not self.topology.has_edge(node1, node2):
                raise ValueError(f"Edge ({node1}, {node2}) does not exist in topology")
            
            # Get index of directed link
            link = self.topology[node1][node2]
            link_idx = link["index"]
            offset = link_idx + band_offset_multiplier
            
            # FIX: Validate that slots are actually free before allocating
            allocation_slice = slice(initial_slot_shift, initial_slot_shift + number_slots)
            
            # Check available_slots matches spectrum_slots_allocation
            if np.any(available_slots[offset, allocation_slice] == 0):
                raise ValueError(
                    f"Attempting to allocate occupied slots on link ({node1}, {node2}), "
                    f"band {band}, slots [{initial_slot_shift}:{initial_slot_shift + number_slots}]. "
                    f"Service {current_service.service_id}"
                )
            
            # Check that spectrum_slots_allocation shows slots as free (-1)
            existing_allocations = self.spectrum_slots_allocation[offset, allocation_slice]
            occupied_slots = existing_allocations[existing_allocations != -1]
            
            if len(occupied_slots) > 0:
                conflicting_services = np.unique(occupied_slots)
                raise ValueError(
                    f"Double allocation detected! Service {current_service.service_id} "
                    f"attempting to use slots already allocated to service(s) {conflicting_services} "
                    f"on link ({node1}, {node2}), band {band}, slots [{initial_slot_shift}:{initial_slot_shift + number_slots}]"
                )
            
            # Update spectrum availability
            available_slots[offset, allocation_slice] = 0
            
            # FIX: Track service ID for validation
            self.spectrum_slots_allocation[offset, allocation_slice] = current_service.service_id
            
            # Update link service lists
            link["services"].append(current_service)
            link["running_services"].append(current_service)

        # Update service and network information
        self.topology.graph["running_services"].append(current_service)
        current_service.path = path
        current_service.band = band
        current_service.initial_slot = initial_slot_shift
        current_service.number_slots = number_slots
        current_service.bandwidth = number_slots * 12.5e9
        current_service.center_frequency = self._calculate_center_frequency(self.current_service)
        
        # Update counters
        self.services_accepted += 1
        self.episode_services_accepted += 1
        
        # Update network fragmentation after allocation
        self._update_network_fragmentation()

    def reward(self, band, path_selected):
        """
        Curriculum-based multi-component reward function.
        
        Stage 1 (0-3000 episodes): R = base + routing
        Stage 2 (3000-8000 episodes): R = base + routing + band  
        Stage 3 (8000+ episodes): R = base + routing + band + spectrum
        
        Args:
            band: Selected band ID
            path_selected: Selected path object
            
        Returns:
            float: Total reward
        """
        
        # ========================================
        # BASE REWARD
        # ========================================
        if self.current_service.accepted:
            R_base = 1.0
        else:
            # No bonuses for rejected services
            self._log_reward_components(
                base=-1.0, routing=0, band=0, spectrum=0, total=-1.0
            )
            return -1.0
        
        # ========================================
        # COMPONENT 1: ROUTING QUALITY BONUS
        # ========================================
        B_routing = self._calculate_routing_bonus(path_selected)
        
        # ========================================
        # COMPONENT 2: BAND SELECTION BONUS
        # ========================================
        B_band = self._calculate_band_bonus(band)
        
        # ========================================
        # COMPONENT 3: SPECTRUM ALLOCATION BONUS
        # ========================================
        B_spectrum = self._calculate_spectrum_bonus(band)
        
        # ========================================
        # APPLY CURRICULUM STAGE
        # ========================================
        if not self.use_curriculum:
            # No curriculum - use full reward
            R_total = R_base + B_routing + B_band + B_spectrum
        else:
            stage = self._get_curriculum_stage()
            
            if stage == 1:
                # Stage 1: Routing only
                R_total = R_base + B_routing
            elif stage == 2:
                # Stage 2: Routing + Band
                R_total = R_base + B_routing + B_band
            else:
                # Stage 3: Full reward
                R_total = R_base + B_routing + B_band + B_spectrum
        
        # Log components for analysis
        self._log_reward_components(
            base=R_base, routing=B_routing, band=B_band, 
            spectrum=B_spectrum, total=R_total
        )
        
        return R_total


# ============================================================================
# ADD THESE HELPER METHODS TO RMSAEnv
# ============================================================================

    def _calculate_routing_bonus(self, path):
        if path is None:
            return 0.0
        
        available_slots = self.get_available_slots(path, band=0)
        initial_indices, values, lengths = RMSAEnv.rle(available_slots)
        
        if len(lengths) > 0:
            available_indices = np.where(values == 1)[0]
            if len(available_indices) > 0:
                F_contiguous = np.max(lengths[available_indices])
            else:
                F_contiguous = 0
        else:
            F_contiguous = 0
        
        F_required = self.current_service.number_slots
        
        if F_required == 0 or F_contiguous == 0:
            return 0.0
        
        max_route_length = 4000
        d_norm = path.length / max_route_length
        
        # FIX: Cap at 1.0
        spectrum_factor = min(1.0, F_contiguous / F_required)
        
        distance_penalty = np.exp(-self.lambda_distance * d_norm)
        
        B_routing = self.w_routing * spectrum_factor * distance_penalty
        
        if self.adaptive_curriculum:
            self.recent_routing_bonuses.append(B_routing)
            if len(self.recent_routing_bonuses) > 500:
                self.recent_routing_bonuses.pop(0)
        
        return B_routing

    def _calculate_band_bonus(self, band):
        """
        Calculate band selection bonus.
        
        B_band = +w_b if correct band, -w_b if wrong band
        
        Rules:
        - High modulation (ML > ML_avg) → C-band
        - Low modulation (ML ≤ ML_avg) → L-band
        """
        if not hasattr(self.current_service, 'modulation_format'):
            return 0.0
        
        # Get modulation level
        ML_selected = self._get_modulation_level(
            self.current_service.modulation_format
        )
        
        # Check if band assignment is correct
        correct_band = (
            (ML_selected > self.ml_avg and band == self.BAND_CONFIG['C_BAND']['id']) or
            (ML_selected <= self.ml_avg and band == self.BAND_CONFIG['L_BAND']['id'])
        )
        
        B_band = self.w_band if correct_band else -self.w_band
        
        # Track for adaptive curriculum
        if self.adaptive_curriculum:
            self.recent_band_accuracies.append(1 if correct_band else 0)
            if len(self.recent_band_accuracies) > 500:
                self.recent_band_accuracies.pop(0)
        
        return B_band


    def _calculate_spectrum_bonus(self, band):
        """
        Calculate spectrum allocation bonus based on FF/LF policy.
        
        Opposite policies in C and L bands:
        - C-band: Large services (F > F_avg) use LF, small use FF
        - L-band: Large services use FF, small use LF
        
        Rewards placement closer to the intended position.
        """
        if not hasattr(self.current_service, 'initial_slot'):
            return 0.0
        
        F_required = self.current_service.number_slots
        initial_slot_global = self.current_service.initial_slot
        
        # Get band range
        if band == self.BAND_CONFIG['C_BAND']['id']:
            band_start = self.BAND_CONFIG['C_BAND']['start_slot']
            band_end = self.BAND_CONFIG['C_BAND']['end_slot']
        else:  # L-band
            band_start = self.BAND_CONFIG['L_BAND']['start_slot']
            band_end = self.BAND_CONFIG['L_BAND']['end_slot']
        
        # Convert to band-local coordinates
        initial_slot_local = initial_slot_global - band_start
        band_size = band_end - band_start + 1
        band_center_local = band_size / 2
        
        # Centrality metric: 1 = center, 0 = edge
        P_centrality = 1 - abs(initial_slot_local - band_center_local) / band_center_local
        P_centrality = max(0, min(1, P_centrality))  # Clamp to [0, 1]
        
        # Apply opposite FF/LF policies
        if band == self.BAND_CONFIG['C_BAND']['id']:
            if F_required > self.f_avg:
                # Large services: LF policy (reward centrality)
                B_spectrum = self.w_spectrum * P_centrality
            else:
                # Small services: FF policy (reward edge placement)
                B_spectrum = self.w_spectrum * (1 - P_centrality)
        else:  # L-band (opposite policies)
            if F_required > self.f_avg:
                # Large services: FF policy (reward edge placement)
                B_spectrum = self.w_spectrum * (1 - P_centrality)
            else:
                # Small services: LF policy (reward centrality)
                B_spectrum = self.w_spectrum * P_centrality
        
        return B_spectrum


    def _get_modulation_level(self, modulation_format):
        """Map modulation format string to level (1-4)."""
        levels = {
            'PM_BPSK': 1,
            'PM_QPSK': 2,
            'PM_8QAM': 3,
            'PM_16QAM': 4
        }
        return levels.get(modulation_format, 0)


    def _get_curriculum_stage(self):
        """Determine current curriculum stage."""
        
        if self.adaptive_curriculum:
            return self._get_adaptive_curriculum_stage()
        else:
            # Fixed episode thresholds
            if self.total_episodes < self.curriculum_stage_1_episodes:
                return 1
            elif self.total_episodes < self.curriculum_stage_2_episodes:
                return 2
            else:
                return 3


    def _get_adaptive_curriculum_stage(self):
        """Adaptive stage progression based on performance."""
        
        if len(self.recent_routing_bonuses) < 500:
            return 1
        
        avg_routing = np.mean(self.recent_routing_bonuses)
        band_accuracy = np.mean(self.recent_band_accuracies) if self.recent_band_accuracies else 0
        
        # Progress when current skill mastered
        if self.current_curriculum_stage == 1:
            if avg_routing >= self.routing_mastery_threshold:
                self.current_curriculum_stage = 2
                print(f"\n{'='*60}")
                print(f"CURRICULUM STAGE 1→2 at episode {self.total_episodes}")
                print(f"Routing mastery: {avg_routing:.3f}")
                print(f"{'='*60}\n")
        
        elif self.current_curriculum_stage == 2:
            if band_accuracy >= self.band_accuracy_threshold:
                self.current_curriculum_stage = 3
                print(f"\n{'='*60}")
                print(f"CURRICULUM STAGE 2→3 at episode {self.total_episodes}")
                print(f"Band accuracy: {band_accuracy*100:.1f}%")
                print(f"{'='*60}\n")
        
        return self.current_curriculum_stage


    def _log_reward_components(self, base, routing, band, spectrum, total):
        """Log reward components for analysis."""
        self.reward_components_history.append({
            'episode': self.total_episodes,
            'stage': self._get_curriculum_stage() if self.use_curriculum else 3,
            'base': base,
            'routing': routing,
            'band': band,
            'spectrum': spectrum,
            'total': total
        })
        
        # Keep only recent history (last 10000 services)
        if len(self.reward_components_history) > 10000:
            self.reward_components_history.pop(0)


    def get_curriculum_statistics(self, last_n=500):
        """Get statistics about curriculum learning progress."""
        if len(self.reward_components_history) < last_n:
            recent = self.reward_components_history
        else:
            recent = self.reward_components_history[-last_n:]
        
        accepted = [r for r in recent if r['base'] > 0]
        
        if not accepted:
            return {'error': 'No accepted services in recent history'}
        
        return {
            'total_episodes': self.total_episodes,
            'current_stage': self._get_curriculum_stage(),
            'acceptance_rate': len(accepted) / len(recent),
            'avg_routing_bonus': np.mean([r['routing'] for r in accepted]),
            'routing_mastery_%': np.mean([r['routing'] for r in accepted]) / self.w_routing * 100,
            'band_accuracy': np.mean([r['band'] > 0 for r in accepted]),
            'avg_spectrum_bonus': np.mean([r['spectrum'] for r in accepted]),
            'avg_total_reward': np.mean([r['total'] for r in accepted])
        }


    def _calculate_f_avg(self):
        """
        Calculate average required frequency slots (F_avg) across all 
        modulation-bandwidth combinations.
        
        F_avg = (1/(|M| × N_B)) × Σ_modulations Σ_bandwidths [⌈B/capacity⌉ + 1]
        
        Returns:
            int: Average number of slots (rounded up)
        """
        if not hasattr(self, 'modulations') or not self.modulations:
            return 4  # Fallback default
        
        if not hasattr(self, 'bit_rates') or not self.bit_rates:
            return 4  # Fallback default
        
        total_slots = 0
        count = 0
        
        for modulation in self.modulations:
            mod_capacity = modulation['capacity']
            
            for bit_rate in self.bit_rates:
                # Calculate slots needed: ceil(bit_rate / capacity) + guard_band
                slots_needed = math.ceil(bit_rate / mod_capacity) + 1
                total_slots += slots_needed
                count += 1
        
        # Return average, rounded up
        return int(math.ceil(total_slots / count)) if count > 0 else 4
    

    def _calculate_ml_avg(self):
        """
        Calculate average modulation level (ML_avg) across all available 
        modulation formats.
        
        ML_avg = (1/|M|) × Σ ML_i
        
        Modulation levels are assigned based on order:
        - First modulation (lowest capacity): ML = 1
        - Second modulation: ML = 2
        - ...
        - Nth modulation (highest capacity): ML = N
        
        Returns:
            int: Average modulation level (floored for balanced classification)
        """
        if not hasattr(self, 'modulations') or not self.modulations:
            return 2  # Fallback default
        
        num_modulations = len(self.modulations)
        
        # Modulation levels: 1, 2, 3, ..., N
        modulation_levels = list(range(1, num_modulations + 1))
        
        # Calculate average
        avg_ml = sum(modulation_levels) / num_modulations
        
        # Floor for balanced classification
        # Example: [1,2,3,4] → avg=2.5 → floor=2
        # This gives: ML≤2 (BPSK,QPSK)→L-band, ML>2 (8QAM,16QAM)→C-band
        return int(np.floor(avg_ml))


    def _release_path(self, service: Service):

        """Modified to validate deallocation integrity"""
        
        for i in range(len(service.path.node_list) - 1):
            node1, node2 = service.path.node_list[i], service.path.node_list[i + 1]
            
            # Validate edge exists
            if not self.topology.has_edge(node1, node2):
                self.logger.warning(
                    f"Attempting to release service {service.service_id} on non-existent "
                    f"edge ({node1}, {node2})"
                )
                continue
            
            # Get index of directed link
            link_idx = self.topology[node1][node2]["index"]
            offset = link_idx + (self.topology.number_of_edges() * service.band)
            
            allocation_slice = slice(service.initial_slot, service.initial_slot + service.number_slots)
            
            # FIX: Validate that the service ID matches before releasing
            allocated_services = self.spectrum_slots_allocation[offset, allocation_slice]
            
            # Check for mismatches
            mismatched_slots = allocated_services[allocated_services != service.service_id]
            if len(mismatched_slots) > 0:
                conflicting_ids = np.unique(mismatched_slots[mismatched_slots != -1])
                self.logger.error(
                    f"Service ID mismatch during release! Service {service.service_id} "
                    f"attempting to release slots allocated to service(s) {conflicting_ids} "
                    f"on link ({node1}, {node2}), band {service.band}, "
                    f"slots [{service.initial_slot}:{service.initial_slot + service.number_slots}]"
                )
                # Continue with release but log the error
            
            # Release spectrum
            self.topology.graph["available_slots"][offset, allocation_slice] = 1
            
            # Clear allocation
            self.spectrum_slots_allocation[offset, allocation_slice] = -1
            
            # Update service lists
            try:
                self.topology[node1][node2]["running_services"].remove(service)
            except ValueError:
                self.logger.warning(
                    f"Service {service.service_id} not found in running_services "
                    f"for link ({node1}, {node2})"
                )
        
        # Remove from global running services
        try:
            self.topology.graph["running_services"].remove(service)
        except ValueError:
            self.logger.warning(
                f"Service {service.service_id} not found in global running_services"
            )
        
        # Update network fragmentation after deallocation
        self._update_network_fragmentation()