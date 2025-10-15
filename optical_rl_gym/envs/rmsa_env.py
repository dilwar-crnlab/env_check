import copy
import functools
import heapq
import logging
import math
from collections import defaultdict
from typing import Optional, Sequence, Tuple

from typing import Any
import gym
import networkx as nx
import numpy as np
import random
from optical_rl_gym.utils import Path, Service
from optical_rl_gym.osnr_calculator import *

from .optical_network_env import OpticalNetworkEnv
from optical_rl_gym.logging_utils import ServiceRequestLogger, LoggingConfig


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
        j=1,
        enable_logging=True, log_config=None, log_dir='logs'
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

        # Initialize logger
        self.enable_logging = enable_logging
        if self.enable_logging:
            config = log_config or LoggingConfig(detail_level='medium')
            self.request_logger = ServiceRequestLogger(
                log_dir=log_dir,
                config=config,
                experiment_name=None  # Can set experiment name
            )
        else:
            self.request_logger = None

        # ADD THIS: Store bit rates and probabilities
        self.bit_rates = list(bit_rates)
        


        self.j = j
        self.bit_rates = list(bit_rates)
        self.min_bit_rate = min(bit_rates)     # Minimum bit rate in Gbps
        self.max_bit_rate = max(bit_rates)    # Maximum bit rate in Gbps
        
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
        self.episode_bit_rate_requested=0
        self.episode_bit_rate_accepted = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0

        # Initialize network fragmentation tracking
        self.topology.graph["current_network_fragmentation"] = 0.0
        self.topology.graph["current_band_fragmentation"] = {band: 0.0 for band in range(self.num_bands)}
        self.topology.graph["previous_network_fragmentation"] = 0.0
        
        # Initialize OSNR tracking dictionaries
        self._affected_services_osnr_before = {}
        self._affected_services_osnr_after = {}



        
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

        # STUR (Spectrum Time Utilization Ratio) tracking
        self.slot_occupation_durations = np.zeros(
            (self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources),
            dtype=float
        )
        self.slot_allocation_times = np.full(
            (self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources),
            fill_value=-1.0,
            dtype=float
        )
        self.episode_start_time = 0.0



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
        
        # Action space for the AGENT (this still respects allow_rejection)
        self.action_space = gym.spaces.MultiDiscrete(
            (
                self.k_paths + self.reject_action,      # Agent can only reject if allowed
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

    # Modulation format
    #Ref. Paper "ISRS impact-reduced routing, modulation, band, and spectrum allocation algorithm in 
    # C + L-bands elastic optical networks"
    # [BPSK, QPSK, 8QAM, 16QAM]
    #capacity = [12.5, 25, 37.5, 50]
    capacity = [50, 100, 150, 200]
    modulations = list()
    modulations.append({'modulation': 'PM_BPSK', 'capacity': capacity[0], 'max_reach': 4000})
    modulations.append({'modulation': 'PM_QPSK', 'capacity': capacity[1], 'max_reach': 2000})
    modulations.append({'modulation': 'PM_8QAM', 'capacity': capacity[2], 'max_reach': 1000})
    modulations.append({'modulation': 'PM_16QAM', 'capacity': capacity[3], 'max_reach': 500})


    def _calculate_center_frequency(self, service: Service) -> float:
        """
        Calculate center frequency using local slots considering slot center offset.
        """
        local_start_slot = service.initial_slot
        local_end_slot = local_start_slot + service.number_slots - 1
        
        # Compute local center slot index (fractional)
        local_center_slot = (local_start_slot + local_end_slot) / 2.0
        #print("local_center_slot",local_center_slot)
        
        band_base_frequency = self.band_frequencies[service.band]['start']
        slot_width_thz = self.channel_width * 1e-3  # Convert GHz to THz
        
        # Add half slot width offset to get center frequency within slot 
        center_freq = band_base_frequency + (local_center_slot + 0.5) * slot_width_thz * 1e12  # in Hz
        
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
    
   
   # def step(self, action: [int]):
        path, band, initial_slot = action[0], action[1], action[2]

        # LOG: Action received
        if self.request_logger:
            self.request_logger.log_action_received(
                service_id=self.current_service.service_id,
                path_idx=path,
                band=band,
                initial_slot=initial_slot
            )

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
                        temp_service.initial_slot = initial_slot
                        temp_service.global_initial_slot = initial_slot_shift  

                        temp_service.number_slots = slots
                        temp_service.path = temp_path
                        
                        
                        center_frequency=self._calculate_center_frequency(temp_service)
                        temp_service.center_frequency = center_frequency
                        #print("temp centerfreq", temp_service.center_frequency)
                        
                        # Get modulation format
                        modulation = self.get_modulation_format(temp_path, self.num_bands, band, self.modulations)['modulation']
                        temp_service.modulation_format = modulation
                        
                        # First check OSNR for the new service (SNR condition check)
                        osnr_db = self.osnr_calculator.calculate_osnr(temp_service, self.topology)
                        #print("Temp service OSNR[dB]", osnr_db, "OSNR th",self.OSNR_th[temp_service.modulation_format])
                        
                        
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
                                self.current_service.center_frequency = center_frequency
                                self.current_service.modulation_format = modulation
                                self.actions_taken[path, band, initial_slot] += 1
                                self._add_release(self.current_service)
                                
                                # Update acceptance counters
                                self.blocking_reasons['total_accepted'] += 1
                                self.episode_blocking_reasons['total_accepted'] += 1
                                
                                # Update bit rate acceptance counters
                                self.bit_rate_accepted += current_bit_rate
                                self.episode_bit_rate_accepted += current_bit_rate
                                #print("Service:", self.current_service)
        
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


    def step(self, action: [int]):
        """
        Execute action and allocate service.
        
        Validates:
            1. Action validity
            2. Path length
            3. Modulation availability
            4. Spectrum availability
            5. OSNR threshold
            6. OSNR interference
        
        Logs:
            - Action received
            - Available paths
            - Each validation step (pass/fail with details)
            - Final outcome (acceptance or rejection with reason)
            - Network state after allocation
        
        Args:
            action: [path_index, band, initial_slot]
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        path, band, initial_slot = action[0], action[1], action[2]
        self.actions_output[path, band, initial_slot] += 1
        
        # ========================================================================
        # LOGGING: Action Received
        # ========================================================================
        if self.request_logger:
            self.request_logger.log_action_received(
                service_id=self.current_service.service_id,
                path_idx=path,
                band=band,
                initial_slot=initial_slot
            )
            
            # Log available paths
            k_paths = self.k_shortest_paths[
                self.current_service.source, 
                self.current_service.destination
            ]
            self.request_logger.log_available_paths(
                service_id=self.current_service.service_id,
                paths=k_paths,
                selected_idx=path
            )
        
        # Track bit rate for current service
        current_bit_rate = self.current_service.bit_rate
        self.bit_rate_requested += current_bit_rate
        self.episode_bit_rate_requested += current_bit_rate
        
        # Start with service rejected
        self.current_service.accepted = False
        blocking_reason = None
        
        # ========================================================================
        # VALIDATION 1: Action Validity Check
        # ========================================================================
        if not (path < self.k_paths and band < self.num_bands and initial_slot < self.num_spectrum_resources):
            blocking_reason = 'invalid_action'
            
            # LOGGING: Validation 1 Failed
            if self.request_logger:
                self.request_logger.log_validation_step(
                    service_id=self.current_service.service_id,
                    step_name="ACTION_VALIDITY_CHECK",
                    passed=False,
                    details={
                        'path_idx': path,
                        'k_paths': self.k_paths,
                        'path_valid': path < self.k_paths,
                        'band': band,
                        'num_bands': self.num_bands,
                        'band_valid': band < self.num_bands,
                        'initial_slot': initial_slot,
                        'num_spectrum_resources': self.num_spectrum_resources,
                        'slot_valid': initial_slot < self.num_spectrum_resources
                    }
                )
        else:
            # LOGGING: Validation 1 Passed
            if self.request_logger:
                self.request_logger.log_validation_step(
                    service_id=self.current_service.service_id,
                    step_name="ACTION_VALIDITY_CHECK",
                    passed=True,
                    details={
                        'path_idx': path,
                        'band': band,
                        'initial_slot': initial_slot
                    }
                )
            
            # Get the selected path
            temp_path = self.k_shortest_paths[
                self.current_service.source, 
                self.current_service.destination
            ][path]
            
            # ====================================================================
            # VALIDATION 2: Path Length Check
            # ====================================================================
            max_reach = max(m['max_reach'] for m in self.modulations)
            
            if temp_path.length > max_reach:
                blocking_reason = 'path_length_exceeded'
                
                # LOGGING: Validation 2 Failed
                if self.request_logger:
                    self.request_logger.log_validation_step(
                        service_id=self.current_service.service_id,
                        step_name="PATH_LENGTH_CHECK",
                        passed=False,
                        details={
                            'path_length': temp_path.length,
                            'max_reach': max_reach,
                            'path': str(temp_path.node_list),
                            'exceeds_by': temp_path.length - max_reach,
                            'available_modulations': [m['modulation'] for m in self.modulations]
                        }
                    )
            else:
                # LOGGING: Validation 2 Passed
                if self.request_logger:
                    self.request_logger.log_validation_step(
                        service_id=self.current_service.service_id,
                        step_name="PATH_LENGTH_CHECK",
                        passed=True,
                        details={
                            'path_length': temp_path.length,
                            'max_reach': max_reach,
                            'path': str(temp_path.node_list)
                        }
                    )
                
                # ================================================================
                # VALIDATION 3: Modulation & Slot Calculation Check
                # ================================================================
                slots = self.get_number_slots(temp_path, self.num_bands, band, self.modulations)
                
                if slots <= 0:
                    blocking_reason = 'no_modulation_available'
                    
                    # LOGGING: Validation 3 Failed
                    if self.request_logger:
                        self.request_logger.log_validation_step(
                            service_id=self.current_service.service_id,
                            step_name="MODULATION_SLOT_CHECK",
                            passed=False,
                            details={
                                'path_length': temp_path.length,
                                'bit_rate': self.current_service.bit_rate,
                                'slots_calculated': slots,
                                'reason': 'No modulation format available for this path length'
                            }
                        )
                else:
                    # Get modulation format
                    modulation = self.get_modulation_format(temp_path, self.num_bands, band, self.modulations)
                    
                    # LOGGING: Validation 3 Passed
                    if self.request_logger:
                        self.request_logger.log_validation_step(
                            service_id=self.current_service.service_id,
                            step_name="MODULATION_SLOT_CHECK",
                            passed=True,
                            details={
                                'path_length': temp_path.length,
                                'selected_modulation': modulation['modulation'],
                                'modulation_capacity': modulation['capacity'],
                                'modulation_max_reach': modulation['max_reach'],
                                'required_slots': slots,
                                'bit_rate': self.current_service.bit_rate
                            }
                        )
                    
                    # ============================================================
                    # VALIDATION 4: Spectrum Availability Check
                    # ============================================================
                    if not self.is_path_free(temp_path, initial_slot, slots, band):
                        blocking_reason = 'spectrum_unavailable'
                        
                        # LOGGING: Validation 4 Failed
                        if self.request_logger:
                            spectrum_details = self._get_spectrum_availability_details(
                                temp_path, initial_slot, slots, band
                            )
                            self.request_logger.log_validation_step(
                                service_id=self.current_service.service_id,
                                step_name="SPECTRUM_AVAILABILITY_CHECK",
                                passed=False,
                                details=spectrum_details
                            )
                    else:
                        # LOGGING: Validation 4 Passed
                        if self.request_logger:
                            spectrum_details = self._get_spectrum_availability_details(
                                temp_path, initial_slot, slots, band
                            )
                            self.request_logger.log_validation_step(
                                service_id=self.current_service.service_id,
                                step_name="SPECTRUM_AVAILABILITY_CHECK",
                                passed=True,
                                details=spectrum_details
                            )
                        
                        # ========================================================
                        # VALIDATION 5: OSNR Threshold Check
                        # ========================================================
                        # Create temporary service for OSNR calculations
                        x = self.get_shift(band)[0]
                        initial_slot_shift = initial_slot + x
                        temp_service = copy.deepcopy(self.current_service)
                        temp_service.bandwidth = slots * 12.5e9  # in Hz
                        temp_service.band = band
                        temp_service.initial_slot = initial_slot_shift
                        temp_service.number_slots = slots
                        temp_service.path = temp_path
                        temp_service.center_frequency = self._calculate_center_frequency(temp_service)
                        
                        # Get modulation format
                        temp_service.modulation_format = modulation['modulation']
                        
                        # Calculate OSNR (logging happens inside osnr_calculator)
                        osnr_db = self.osnr_calculator.calculate_osnr(temp_service, self.topology)
                        
                        if osnr_db < self.OSNR_th[temp_service.modulation_format]:
                            blocking_reason = 'osnr_threshold_violation'
                            
                            # LOGGING: Validation 5 Failed
                            if self.request_logger:
                                self.request_logger.log_validation_step(
                                    service_id=self.current_service.service_id,
                                    step_name="OSNR_THRESHOLD_CHECK",
                                    passed=False,
                                    details={
                                        'calculated_osnr': osnr_db,
                                        'required_osnr': self.OSNR_th[temp_service.modulation_format],
                                        'osnr_shortfall': osnr_db - self.OSNR_th[temp_service.modulation_format],
                                        'modulation': temp_service.modulation_format,
                                        'center_frequency_thz': temp_service.center_frequency / 1e12,
                                        'bandwidth_ghz': temp_service.bandwidth / 1e9,
                                        'path': str(temp_path.node_list)
                                    }
                                )
                        else:
                            # LOGGING: Validation 5 Passed
                            if self.request_logger:
                                self.request_logger.log_validation_step(
                                    service_id=self.current_service.service_id,
                                    step_name="OSNR_THRESHOLD_CHECK",
                                    passed=True,
                                    details={
                                        'calculated_osnr': osnr_db,
                                        'required_osnr': self.OSNR_th[temp_service.modulation_format],
                                        'osnr_margin': osnr_db - self.OSNR_th[temp_service.modulation_format],
                                        'modulation': temp_service.modulation_format,
                                        'center_frequency_thz': temp_service.center_frequency / 1e12,
                                        'bandwidth_ghz': temp_service.bandwidth / 1e9
                                    }
                                )
                            
                            # ====================================================
                            # VALIDATION 6: OSNR Interference Check
                            # ====================================================
                            if not self._check_existing_services_osnr(temp_path, band, temp_service):
                                blocking_reason = 'osnr_interference_violation'
                                
                                # LOGGING: Validation 6 Failed
                                if self.request_logger:
                                    affected_details = self._get_affected_services_details()
                                    
                                    # Find which services would fail
                                    failing_services = [s for s in affected_details if s['would_fail']]
                                    
                                    self.request_logger.log_validation_step(
                                        service_id=self.current_service.service_id,
                                        step_name="OSNR_INTERFERENCE_CHECK",
                                        passed=False,
                                        details={
                                            'num_affected_services': len(affected_details),
                                            'num_failing_services': len(failing_services),
                                            'affected_services': affected_details,
                                            'reason': 'Would cause existing services to drop below OSNR threshold'
                                        }
                                    )
                            else:
                                # LOGGING: Validation 6 Passed
                                if self.request_logger:
                                    affected_details = self._get_affected_services_details()
                                    self.request_logger.log_validation_step(
                                        service_id=self.current_service.service_id,
                                        step_name="OSNR_INTERFERENCE_CHECK",
                                        passed=True,
                                        details={
                                            'num_affected_services': len(affected_details),
                                            'affected_services': affected_details,
                                            'all_services_ok': True,
                                            'note': 'All affected services maintain acceptable OSNR'
                                        }
                                    )
                                
                                # ================================================
                                # ALL VALIDATIONS PASSED - ACCEPT SERVICE
                                # ================================================
                                self.current_service.current_OSNR = osnr_db
                                self.current_service.OSNR_th = self.OSNR_th[temp_service.modulation_format]
                                self.current_service.OSNR_margin = self.current_service.current_OSNR - self.current_service.OSNR_th
                                
                                # Provision the path
                                self._provision_path(temp_path, initial_slot, slots, band, self.current_service.arrival_time)
                                self.current_service.accepted = True
                                self.current_service.modulation_format = modulation['modulation']
                                self.actions_taken[path, band, initial_slot] += 1
                                self._add_release(self.current_service)
                                
                                # Update acceptance counters
                                self.blocking_reasons['total_accepted'] += 1
                                self.episode_blocking_reasons['total_accepted'] += 1
                                
                                # Update bit rate acceptance counters
                                self.bit_rate_accepted += current_bit_rate
                                self.episode_bit_rate_accepted += current_bit_rate
                                
                                # ============================================
                                # LOGGING: Service Accepted
                                # ============================================
                                if self.request_logger:
                                    self.request_logger.log_service_accepted(
                                        service_id=self.current_service.service_id,
                                        path=temp_path,
                                        band=band,
                                        initial_slot=initial_slot_shift,
                                        number_slots=slots,
                                        modulation=modulation['modulation'],
                                        osnr_db=osnr_db,
                                        osnr_margin=self.current_service.OSNR_margin,
                                        center_frequency=self.current_service.center_frequency / 1e12,
                                        bandwidth=self.current_service.bandwidth / 1e9,
                                        links_used=[f"{temp_path.node_list[i]}→{temp_path.node_list[i+1]}" 
                                                for i in range(len(temp_path.node_list)-1)]
                                    )
                                    
                                    # LOGGING: Network State After Allocation
                                    self._log_network_state_after_allocation(temp_path, band)
        
        # ========================================================================
        # HANDLE REJECTION
        # ========================================================================
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
                
                # ================================================================
                # LOGGING: Service Rejected
                # ================================================================
                if self.request_logger:
                    self.request_logger.log_service_rejected(
                        service_id=self.current_service.service_id,
                        blocking_reason=blocking_reason,
                        attempted_path=temp_path if 'temp_path' in locals() else None,
                        attempted_band=band,
                        attempted_slot=initial_slot,
                        details=self._get_rejection_details(blocking_reason)
                    )
        
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

        # Create info dictionary with metrics
        info = {
            "band": band if self.services_accepted else -1,
            "service_blocking_rate": (self.services_processed - self.services_accepted) / self.services_processed,
            "episode_service_blocking_rate": (self.episode_services_processed - self.episode_services_accepted) / self.episode_services_processed,
            "bit_rate_blocking_rate": bit_rate_blocking_rate,
            "episode_bit_rate_blocking_rate": episode_bit_rate_blocking_rate,
            "blocking_reason": blocking_reason if blocking_reason else "accepted",
            "blocking_reasons_count": self.blocking_reasons.copy(),
            "episode_blocking_reasons_count": self.episode_blocking_reasons.copy(),
            
            # REPLACE: Use STUR methods instead
            "network_utilization": self._calculate_network_stur(),  # Changed
            "band_utilization": self._calculate_band_stur(),        # Changed
            
            "network_fragmentation": self.topology.graph.get("current_network_fragmentation", 0.0),
            "band_fragmentation": self.topology.graph.get("current_band_fragmentation", {}).copy(),
            **blocking_rate_breakdown,
            **episode_blocking_rate_breakdown
        }

        self._new_service = False
        self._next_service()
        
        return (self.observation(), reward, self.episode_services_processed == self.episode_length, info)


    # def reset(self, only_episode_counters=True):
    #     """
    #     Reset environment for new episode.
        
    #     Simple DQN version without curriculum learning.
        
    #     Args:
    #         only_episode_counters: If True, soft reset (between episodes)
    #                             If False, hard reset (full reinitialization)
        
    #     Logs:
    #         - Episode statistics before reset (if soft reset)
        
    #     Returns:
    #         np.array: Initial observation after reset
    #     """
    #     # ========================================================================
    #     # LOG EPISODE STATISTICS (Before Episode Reset)
    #     # ========================================================================
    #     if only_episode_counters and self.request_logger:
    #         # Calculate all statistics for the completed episode
    #         stats = {
    #             'episode': self.episode_services_processed // self.episode_length if self.episode_length > 0 else 0,
    #             'services_processed': self.episode_services_processed,
    #             'services_accepted': self.episode_services_accepted,
    #             'blocking_rate': (
    #                 (self.episode_services_processed - self.episode_services_accepted) / 
    #                 self.episode_services_processed 
    #                 if self.episode_services_processed > 0 else 0
    #             ),
                
    #             # Blocking reasons breakdown
    #             'blocking_reasons': self.episode_blocking_reasons.copy() if hasattr(self, 'episode_blocking_reasons') else {},
                
    #             # Bit rate statistics
    #             'bit_rate_requested': (
    #                 self.episode_bit_rate_requested 
    #                 if hasattr(self, 'episode_bit_rate_requested') else 0
    #             ),
    #             'bit_rate_accepted': (
    #                 self.episode_bit_rate_accepted 
    #                 if hasattr(self, 'episode_bit_rate_accepted') else 0
    #             ),
    #             'bit_rate_blocking_rate': (
    #                 (self.episode_bit_rate_requested - self.episode_bit_rate_accepted) /
    #                 self.episode_bit_rate_requested 
    #                 if hasattr(self, 'episode_bit_rate_requested') and self.episode_bit_rate_requested > 0 
    #                 else 0
    #             ),
                
    #             # Network utilization
    #             'network_utilization': self._calculate_network_utilization(),
    #             'band_utilization': self._calculate_band_utilization(),
                
    #             # OSNR statistics
    #             'avg_osnr_margin': self._calculate_avg_osnr_margin(),
    #             'band_osnr_margins': self._calculate_band_osnr_margins(),
                
    #             # Fragmentation
    #             'network_fragmentation': self.topology.graph.get('current_network_fragmentation', 0.0),
    #             'band_fragmentation': self.topology.graph.get('current_band_fragmentation', {}).copy(),
    #         }
            
    #         # Log episode statistics
    #         self.request_logger.log_episode_statistics(stats)
            
    #         # Flush logger buffer every 10 episodes
    #         if self.services_processed > 0 and (self.services_processed // self.episode_length) % 10 == 0:
    #             self.request_logger.flush()
        
    #     # ========================================================================
    #     # EPISODE-LEVEL RESET (happens every episode)
    #     # ========================================================================
    #     self.episode_services_processed = 0
    #     self.episode_services_accepted = 0
        
    #     # Reset episode bit rate tracking
    #     if hasattr(self, 'episode_bit_rate_requested'):
    #         self.episode_bit_rate_requested = 0
    #     if hasattr(self, 'episode_bit_rate_accepted'):
    #         self.episode_bit_rate_accepted = 0
        
    #     # Reset episode action tracking
    #     self.episode_actions_output = np.zeros(
    #         (
    #             self.k_paths + 1,
    #             self.num_bands + 1,
    #             self.num_spectrum_resources + 1,
    #         ),
    #         dtype=int,
    #     )
    #     self.episode_actions_taken = np.zeros(
    #         (
    #             self.k_paths + 1,
    #             self.num_bands + 1,
    #             self.num_spectrum_resources + 1,
    #         ),
    #         dtype=int,
    #     )
        
    #     # Reset episode blocking reasons
    #     if hasattr(self, 'episode_blocking_reasons'):
    #         self.episode_blocking_reasons = {
    #             'invalid_action': 0,
    #             'path_length_exceeded': 0,
    #             'no_modulation_available': 0,
    #             'spectrum_unavailable': 0,
    #             'osnr_threshold_violation': 0,
    #             'osnr_interference_violation': 0,
    #             'total_blocked': 0,
    #             'total_accepted': 0
    #         }
        
    #     # ========================================================================
    #     # SOFT RESET: Return observation for new episode
    #     # ========================================================================
    #     if only_episode_counters:
    #         # Increment service counter if there's a pending service
    #         if self._new_service:
    #             self.episode_services_processed += 1
            
    #         return self.observation()
        
    #     # ========================================================================
    #     # FULL RESET (happens at training start or explicit reset)
    #     # ========================================================================
    #     super().reset()
        
    #     # Reset cumulative bit rate tracking
    #     if hasattr(self, 'bit_rate_requested'):
    #         self.bit_rate_requested = 0
    #     if hasattr(self, 'bit_rate_accepted'):
    #         self.bit_rate_accepted = 0
        
    #     # Reset cumulative action tracking
    #     self.actions_output = np.zeros(
    #         (
    #             self.k_paths + 1,
    #             self.num_bands + 1,
    #             self.num_spectrum_resources + 1,
    #         ),
    #         dtype=int,
    #     )
    #     self.actions_taken = np.zeros(
    #         (
    #             self.k_paths + 1,
    #             self.num_bands + 1,
    #             self.num_spectrum_resources + 1,
    #         ),
    #         dtype=int,
    #     )
        
    #     # Reset cumulative blocking reasons
    #     if hasattr(self, 'blocking_reasons'):
    #         self.blocking_reasons = {
    #             'invalid_action': 0,
    #             'path_length_exceeded': 0,
    #             'no_modulation_available': 0,
    #             'spectrum_unavailable': 0,
    #             'osnr_threshold_violation': 0,
    #             'osnr_interference_violation': 0,
    #             'total_blocked': 0,
    #             'total_accepted': 0
    #         }
        
    #     # Reset fragmentation tracking
    #     self.topology.graph["current_network_fragmentation"] = 0.0
    #     self.topology.graph["current_band_fragmentation"] = {
    #         band: 0.0 for band in range(self.num_bands)
    #     }
    #     self.topology.graph["previous_network_fragmentation"] = 0.0
        
    #     # Reset OSNR tracking
    #     self._affected_services_osnr_before = {}
    #     self._affected_services_osnr_after = {}
        
    #     # Reset spectrum allocation
    #     num_edges = self.topology.number_of_edges()
    #     self.topology.graph["available_slots"] = np.ones(
    #         (num_edges * self.num_bands, self.num_spectrum_resources), 
    #         dtype=int
    #     )
    #     self.spectrum_slots_allocation = np.full(
    #         (num_edges * self.num_bands, self.num_spectrum_resources),
    #         fill_value=-1, 
    #         dtype=int
    #     )
        
    #     # ========================================================================
    #     # PRINT INITIALIZATION MESSAGE (only on full reset)
    #     # ========================================================================
    #     if self.request_logger:
    #         print(f"\n{'='*70}")
    #         print(f"{'ENVIRONMENT INITIALIZED':^70}")
    #         print(f"{'='*70}")
    #         print(f"Topology: {self.topology.graph['name']}")
    #         print(f"Nodes: {self.topology.number_of_nodes()}")
    #         print(f"Edges: {self.topology.number_of_edges()}")
    #         print(f"K-shortest paths: {self.k_paths}")
    #         print(f"Bands: {self.num_bands} ({'C-band' if self.num_bands == 1 else 'C+L bands'})")
    #         print(f"Spectrum resources: {self.num_spectrum_resources} slots")
    #         print(f"Episode length: {self.episode_length} services")
    #         print(f"Bit rates: {self.bit_rates} Gbps")
    #         print(f"Load: {self.load:.2f} Erlangs")
    #         print(f"Mean service holding time: {self.mean_service_holding_time:.2f} s")
    #         print(f"Logging enabled: {self.enable_logging}")
    #         if self.enable_logging:
    #             print(f"Log directory: {self.request_logger.log_dir}")
    #         print(f"{'='*70}\n")
        
    #     # Generate first service
    #     self._new_service = False
    #     self._next_service()
        
    #     return self.observation()


    def reset(self, only_episode_counters=True):
        """
        Reset environment for new episode.
        
        Simple DQN version without curriculum learning.
        
        Args:
            only_episode_counters: If True, soft reset (between episodes)
                                If False, hard reset (full reinitialization)
        
        Logs:
            - Episode statistics before reset (if soft reset)
        
        Returns:
            np.array: Initial observation after reset
        """
        # ========================================================================
        # LOG EPISODE STATISTICS (Before Episode Reset)
        # ========================================================================
        if only_episode_counters and self.request_logger:
            # Calculate all statistics for the completed episode
            stats = {
                'episode': self.episode_services_processed // self.episode_length if self.episode_length > 0 else 0,
                'services_processed': self.episode_services_processed,
                'services_accepted': self.episode_services_accepted,
                'blocking_rate': (
                    (self.episode_services_processed - self.episode_services_accepted) / 
                    self.episode_services_processed 
                    if self.episode_services_processed > 0 else 0
                ),
                
                # Blocking reasons breakdown
                'blocking_reasons': self.episode_blocking_reasons.copy() if hasattr(self, 'episode_blocking_reasons') else {},
                
                # Bit rate statistics
                'bit_rate_requested': (
                    self.episode_bit_rate_requested 
                    if hasattr(self, 'episode_bit_rate_requested') else 0
                ),
                'bit_rate_accepted': (
                    self.episode_bit_rate_accepted 
                    if hasattr(self, 'episode_bit_rate_accepted') else 0
                ),
                'bit_rate_blocking_rate': (
                    (self.episode_bit_rate_requested - self.episode_bit_rate_accepted) /
                    self.episode_bit_rate_requested 
                    if hasattr(self, 'episode_bit_rate_requested') and self.episode_bit_rate_requested > 0 
                    else 0
                ),
                
                
                
                # STUR metrics (time-based utilization) - NEW
                'network_utilization': self._calculate_network_stur(),
                'band_utilization': self._calculate_band_stur(),
                
                # OSNR statistics
                'avg_osnr_margin': self._calculate_avg_osnr_margin(),
                'band_osnr_margins': self._calculate_band_osnr_margins(),
                
                # Fragmentation
                'network_fragmentation': self.topology.graph.get('current_network_fragmentation', 0.0),
                'band_fragmentation': self.topology.graph.get('current_band_fragmentation', {}).copy(),
            }
            
            # Log episode statistics
            self.request_logger.log_episode_statistics(stats)
            
            # Flush logger buffer every 10 episodes
            if self.services_processed > 0 and (self.services_processed // self.episode_length) % 10 == 0:
                self.request_logger.flush()
        
        # ========================================================================
        # EPISODE-LEVEL RESET (happens every episode)
        # ========================================================================
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        
        # Reset episode bit rate tracking
        if hasattr(self, 'episode_bit_rate_requested'):
            self.episode_bit_rate_requested = 0
        if hasattr(self, 'episode_bit_rate_accepted'):
            self.episode_bit_rate_accepted = 0
        
        # Reset episode action tracking
        self.episode_actions_output = np.zeros(
            (
                self.k_paths + 1,
                self.num_bands + 1,
                self.num_spectrum_resources + 1,
            ),
            dtype=int,
        )
        self.episode_actions_taken = np.zeros(
            (
                self.k_paths + 1,
                self.num_bands + 1,
                self.num_spectrum_resources + 1,
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
        # EPISODE-LEVEL STUR RESET
        # ========================================================================
        # Note: We keep cumulative STUR tracking across episodes within a simulation
        # Only reset the episode start time for proper episode-level STUR calculation
        if only_episode_counters:
            # Update episode start time for next episode
            self.episode_start_time = self.current_time
        
        # ========================================================================
        # SOFT RESET: Return observation for new episode
        # ========================================================================
        if only_episode_counters:
            # Increment service counter if there's a pending service
            if self._new_service:
                self.episode_services_processed += 1
            
            return self.observation()
        
        # ========================================================================
        # FULL RESET (happens at training start or explicit reset)
        # ========================================================================
        super().reset()
        
        # Reset cumulative bit rate tracking
        if hasattr(self, 'bit_rate_requested'):
            self.bit_rate_requested = 0
        if hasattr(self, 'bit_rate_accepted'):
            self.bit_rate_accepted = 0
        
        # Reset cumulative action tracking
        self.actions_output = np.zeros(
            (
                self.k_paths + 1,
                self.num_bands + 1,
                self.num_spectrum_resources + 1,
            ),
            dtype=int,
        )
        self.actions_taken = np.zeros(
            (
                self.k_paths + 1,
                self.num_bands + 1,
                self.num_spectrum_resources + 1,
            ),
            dtype=int,
        )
        
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
        # FULL STUR RESET
        # ========================================================================
        # Reset all STUR tracking arrays
        self.slot_occupation_durations = np.zeros(
            (self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources),
            dtype=float
        )
        self.slot_allocation_times = np.full(
            (self.topology.number_of_edges() * self.num_bands, self.num_spectrum_resources),
            fill_value=-1.0,
            dtype=float
        )
        
        # Initialize episode start time
        self.episode_start_time = 0.0  # Will be set to current_time after reset
        
        # ========================================================================
        # PRINT INITIALIZATION MESSAGE (only on full reset)
        # ========================================================================
        if self.request_logger:
            print(f"\n{'='*70}")
            print(f"{'ENVIRONMENT INITIALIZED':^70}")
            print(f"{'='*70}")
            print(f"Topology: {self.topology.graph['name']}")
            print(f"Nodes: {self.topology.number_of_nodes()}")
            print(f"Edges: {self.topology.number_of_edges()}")
            print(f"K-shortest paths: {self.k_paths}")
            print(f"Bands: {self.num_bands} ({'C-band' if self.num_bands == 1 else 'C+L bands'})")
            print(f"Spectrum resources: {self.num_spectrum_resources} slots")
            print(f"Episode length: {self.episode_length} services")
            print(f"Bit rates: {self.bit_rates} Gbps")
            print(f"Load: {self.load:.2f} Erlangs")
            print(f"Logging enabled: {self.enable_logging}")
            if self.enable_logging:
                print(f"Log directory: {self.request_logger.log_dir}")
            print(f"STUR tracking: Enabled")
            print(f"{'='*70}\n")
        
        # Generate first service
        self._new_service = False
        self._next_service()
        
        # Set episode start time after initial service generation
        self.episode_start_time = self.current_time
        
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

        bit_rate = self.rng.uniform(self.min_bit_rate, self.max_bit_rate)


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

        # LOG: Service arrival
        if self.request_logger:
            self.request_logger.log_service_arrival(
                service_id=self.current_service.service_id,
                episode=self.total_episodes if hasattr(self, 'total_episodes') else 0,
                source=self.current_service.source,
                destination=self.current_service.destination,
                bit_rate=self.current_service.bit_rate,
                arrival_time=self.current_service.arrival_time,
                holding_time=self.current_service.holding_time
            )

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

            #Added for STUR
            self.slot_allocation_times[offset, allocation_slice] = at
            
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
        current_service.global_initial_slot = initial_slot_shift
        current_service.initial_slot = initial_slot
        current_service.number_slots = number_slots
        current_service.bandwidth = number_slots * 12.5e9 #in Hz
        current_service.center_frequency = self._calculate_center_frequency(self.current_service)
        
        # Update counters
        self.services_accepted += 1
        self.episode_services_accepted += 1
        
        # Update network fragmentation after allocation
        self._update_network_fragmentation()

    def reward(self, band, path_selected):

        if self.current_service.accepted:
            return  1.0
        else:
            return -1.0
           




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
            
            allocation_slice = slice(service.global_initial_slot, service.global_initial_slot + service.number_slots)
            
            # Calculate and accumulate occupation duration for STUR
            allocation_times = self.slot_allocation_times[offset, allocation_slice]
            release_time = self.current_time
            
            for slot_idx in range(len(allocation_times)):
                if allocation_times[slot_idx] >= 0:
                    duration = release_time - allocation_times[slot_idx]
                    actual_slot = service.global_initial_slot + slot_idx
                    self.slot_occupation_durations[offset, actual_slot] += duration

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
            #added for stur
            self.slot_allocation_times[offset, allocation_slice] = -1.0
            
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

    # ====== ADD THESE HELPER METHODS AT END OF RMSAEnv CLASS ======

    def _get_spectrum_availability_details(self, path, initial_slot, slots, band):
        """Get detailed spectrum availability for logging"""
        details = {
            'required_slots': slots,
            'initial_slot_local': initial_slot,
            'band': 'C-band' if band == 0 else 'L-band',
            'band_range': f"{self.get_shift(band)[0]}-{self.get_shift(band)[1]}",
            'global_slot_range': f"{initial_slot + self.get_shift(band)[0]}-"
                            f"{initial_slot + self.get_shift(band)[0] + slots - 1}",
            'link_availability': []
        }
        
        x = self.get_shift(band)[0]
        initial_slot_global = initial_slot + x
        
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i+1]
            link_idx = self.topology[node1][node2]['index']
            offset = link_idx + (self.topology.number_of_edges() * band)
            
            link_spectrum = self.topology.graph['available_slots'][
                offset,
                initial_slot_global:initial_slot_global + slots
            ]
            
            is_available = np.all(link_spectrum == 1)
            
            details['link_availability'].append({
                'link': f"{node1}→{node2}",
                'index': link_idx,
                'available': is_available,
                'occupied_slots': list(np.where(link_spectrum == 0)[0]) if not is_available else []
            })
        
        return details

    def _get_affected_services_details(self):
        """Get details of services affected by OSNR interference"""
        affected = []
        
        if not hasattr(self, '_affected_services_osnr_before'):
            return affected
        
        for service_id, osnr_before in self._affected_services_osnr_before.items():
            osnr_after = self._affected_services_osnr_after.get(service_id)
            
            service = next((s for s in self.topology.graph['running_services'] 
                        if s.service_id == service_id), None)
            
            if service and osnr_after:
                affected.append({
                    'service_id': service_id,
                    'osnr_before': osnr_before,
                    'osnr_after': osnr_after,
                    'osnr_degradation': osnr_before - osnr_after,
                    'threshold': self.OSNR_th.get(service.modulation_format, 0),
                    'margin_before': osnr_before - self.OSNR_th.get(service.modulation_format, 0),
                    'margin_after': osnr_after - self.OSNR_th.get(service.modulation_format, 0),
                    'would_fail': osnr_after < self.OSNR_th.get(service.modulation_format, 0)
                })
        
        return affected

    def _get_rejection_details(self, blocking_reason):
        """Get detailed information about rejection"""
        details = {'reason': blocking_reason}
        
        if blocking_reason == 'spectrum_unavailable':
            details['note'] = "No contiguous spectrum block found"
        elif blocking_reason == 'osnr_threshold_violation':
            details['note'] = "Service OSNR below modulation threshold"
        elif blocking_reason == 'osnr_interference_violation':
            details['note'] = "Would degrade existing services below threshold"
            details['affected_services'] = self._get_affected_services_details()
        elif blocking_reason == 'path_length_exceeded':
            details['note'] = "Path too long for any available modulation"
        elif blocking_reason == 'no_modulation_available':
            details['note'] = "No modulation format supports this path length"
        elif blocking_reason == 'invalid_action':
            details['note'] = "Invalid action indices"
        
        return details

    def _log_network_state_after_allocation(self, path, band):
        """Log network state after allocation"""
        if not self.request_logger:
            return
        
        for i in range(len(path.node_list) - 1):
            node1, node2 = path.node_list[i], path.node_list[i+1]
            link_idx = self.topology[node1][node2]['index']
            
            running_services = self.topology[node1][node2]['running_services']
            
            band_start, band_end = self.get_shift(band)
            spectrum = self.topology.graph['available_slots'][
                link_idx + (self.topology.number_of_edges() * band),
                band_start:band_end
            ]
            
            occupied_slots = np.sum(spectrum == 0)
            total_slots = band_end - band_start
            utilization = occupied_slots / total_slots
            
            self.request_logger.log_link_state(
                link=f"{node1}→{node2}",
                link_index=link_idx,
                band=band,
                num_services=len(running_services),
                occupied_slots=occupied_slots,
                total_slots=total_slots,
                utilization=utilization,
                running_services=[
                    {
                        'service_id': s.service_id,
                        'slots': f"{s.initial_slot}-{s.initial_slot+s.number_slots-1}",
                        'bit_rate': s.bit_rate,
                        'modulation': s.modulation_format if hasattr(s, 'modulation_format') else None,
                        'osnr_db': s.current_OSNR if hasattr(s, 'current_OSNR') else None
                    }
                    for s in running_services
                ]
            )

    def _calculate_network_utilization(self):
        """Calculate overall network utilization"""
        total_slots = self.num_spectrum_resources * self.topology.number_of_edges() * self.num_bands
        available = np.sum(self.topology.graph["available_slots"])
        occupied = total_slots - available
        return occupied / total_slots if total_slots > 0 else 0.0

    def _calculate_band_utilization(self):
        """Calculate per-band utilization"""
        num_edges = self.topology.number_of_edges()
        utilization = {}
        
        for band in range(self.num_bands):
            band_start, band_end = self.get_shift(band)
            band_size = band_end - band_start
            total_slots = band_size * num_edges
            
            available = 0
            for edge_idx in range(num_edges):
                offset = edge_idx + (num_edges * band)
                available += np.sum(self.topology.graph["available_slots"][offset, band_start:band_end])
            
            occupied = total_slots - available
            utilization[band] = occupied / total_slots if total_slots > 0 else 0.0
        
        return utilization

    def _calculate_network_stur(self):
        """
        Calculate Spectrum Time Utilization Ratio (STUR) for entire network.
        
        STUR = Σ(all links) Σ(all slots) [Duration occupied] / 
            (Total Links × Total Slots × Time Period)
        """
        total_occupation_time = np.sum(self.slot_occupation_durations)
        
        # Total available capacity = Links × Bands × Slots × Time Period
        num_links = self.topology.number_of_edges()
        time_period = self.current_time - self.episode_start_time
        
        if time_period <= 0:
            return 0.0
        
        total_capacity = num_links * self.num_bands * self.num_spectrum_resources * time_period
        
        return total_occupation_time / total_capacity if total_capacity > 0 else 0.0

    def _calculate_band_stur(self):
        """
        Calculate Spectrum Time Utilization Ratio (STUR) per band.
        
        Returns:
            dict: STUR value for each band
        """
        num_edges = self.topology.number_of_edges()
        time_period = self.current_time - self.episode_start_time
        
        if time_period <= 0:
            return {band: 0.0 for band in range(self.num_bands)}
        
        stur_per_band = {}
        
        for band in range(self.num_bands):
            band_start, band_end = self.get_shift(band)
            band_size = band_end - band_start
            
            total_occupation = 0.0
            for edge_idx in range(num_edges):
                offset = edge_idx + (num_edges * band)
                total_occupation += np.sum(
                    self.slot_occupation_durations[offset, band_start:band_end]
                )
            
            total_capacity = num_edges * band_size * time_period
            stur_per_band[band] = total_occupation / total_capacity if total_capacity > 0 else 0.0
        
        return stur_per_band

# ==============================================================
