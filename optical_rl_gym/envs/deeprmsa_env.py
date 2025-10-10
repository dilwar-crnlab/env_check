
"""
DeepRMSAEnv: Deep Reinforcement Learning environment for Routing, Modulation, 
Spectrum, and Band Assignment (RMSBA) in elastic optical networks.

This environment extends RMSAEnv to provide:
- Multi-block action space (j blocks per path-band)
- Rich observation space with path/band characteristics
- Statistical normalization of features
- Multi-component reward function (acceptance + fragmentation + OSNR)
"""

from typing import Tuple, Sequence
import copy
import gym
import gym.spaces
import numpy as np

from .rmsa_env import RMSAEnv
from .optical_network_env import OpticalNetworkEnv



class DeepRMSAEnv(RMSAEnv):
    """
    Deep RL environment for multi-band optical network resource allocation.
    
    State Space:
        - Source/destination one-hot encoding (2×|V|)
        - Required slots per path (k)
        - Per path-band features (k×B×(2j+10)):
            * j block starting indices
            * j block sizes
            * Required FSs, avg block size, total available FSs
            * Fragmentation, path metrics, modulation, OSNR margin
    
    Action Space:
        - Discrete actions: k_paths × num_bands × j + reject_action
        - Each action selects (path, band, block) or rejection
    
    Reward:
        - Multi-component: acceptance + fragmentation penalty + OSNR penalty
    """
    
    def __init__(
        self,
        num_bands,
        topology=None,
        j=1,
        episode_length=1000,
        mean_service_holding_time=25.0,
        mean_service_inter_arrival_time=0.1,
        node_request_probabilities=None,
        seed=None,
        bit_rates=[50, 100, 200],
        k_paths=5,
        allow_rejection=False
    ):
        # Initialize parent RMSA environment
        super().__init__(
            num_bands=num_bands,
            topology=topology,
            episode_length=episode_length,
            load=mean_service_holding_time / mean_service_inter_arrival_time,
            mean_service_holding_time=mean_service_holding_time,
            node_request_probabilities=node_request_probabilities,
            seed=seed,
            k_paths=k_paths,
            allow_rejection=allow_rejection,
            reset=False,  # We'll call reset manually after full initialization
        )

        # Action space parameters
        self.j = j  # Number of blocks to consider per path-band
        self.allow_rejection=allow_rejection
        
        
        # Modulation format encoding for observation
        # Higher values = more efficient modulation formats
        self.modulation_encoding = {
            'PM_BPSK': 0.25,   # Lowest spectral efficiency
            'PM_QPSK': 0.50,
            'PM_8QAM': 0.75,
            'PM_16QAM': 1.0    # Highest spectral efficiency
        }

     
        self.bit_rates=bit_rates
        # Calculate observation space dimensions
        num_nodes = self.topology.number_of_nodes()
        path_features_per_path = 7
        band_features_per_path_band = 2 * self.j + 4

        observation_size = (
            1 +                                                       # Bit rate
            2 * num_nodes +                                           # Source + dest
            self.k_paths * path_features_per_path +                   # Path features
            self.k_paths * self.num_bands * band_features_per_path_band  # Band features
        )

        # Use [-2, 2] bounds to accommodate normalized values with margin
        self.observation_space = gym.spaces.Box(
            low=-2**30,
            high=2**30,
            shape=(observation_size,),
            dtype=np.float32
        )

        self.total_actions = self.k_paths * self.num_bands * self.j + self.allow_rejection
        #self.total_actions = self.k_paths * self.num_bands * self.j + 
        # if self.reject_action:
        #     self.total_actions += 1  # Add rejection action

        self.action_space = gym.spaces.Discrete(self.total_actions)

        print(f"\nObservation Space:")
        print(f"  Total size: {observation_size}")
        print(f"  - Bit rate: 1")
        print(f"  - Source/Dest one-hot: {2*num_nodes}")
        print(f"  - Path features: {self.k_paths} × {path_features_per_path} = {self.k_paths * path_features_per_path}")
        print(f"  - Band features: {self.k_paths} × {self.num_bands} × {band_features_per_path_band} = {self.k_paths * self.num_bands * band_features_per_path_band}")

        # Seed spaces for reproducibility
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        # Perform initial reset
        self.reset(only_episode_counters=False)

             

    def observation(self):
        """
        Generate normalized observation using optical_rl_gym style normalization.
        Formula: 2 * (value - 0.5 * (min + max)) / (max - min)
        Maps features to approximately [-1, 1] range.
        
        Observation Structure:
            [bit_rate | source_onehot | dest_onehot | path_features | band_features]
        
        Bit rates: [100, 200, 400] Gbps
        """
        self.min_num_spans=3
        self.max_num_spans=self.topology.graph['max_num_spans']
        self.max_path_length=self.topology.graph['max_path_length']
        self.min_path_length=168
        self.max_num_links=self.topology.graph['max_num_links']
        self.max_avg_link_length=self.topology.graph['avg_link_length']
        self.min_avg_link_length=168

        src = self.current_service.source
        dst = self.current_service.destination
        paths = self.k_shortest_paths[(src, dst)]
        
        # ===================================================================
        # COMPONENT 0: Bit Rate (1 feature)
        # ===================================================================
        current_bit_rate = self.current_service.bit_rate

        
        # Get min and max from environment's bit_rates: [100, 200, 400]
        min_bit_rate = float(min(self.bit_rates))  # 100
        max_bit_rate = float(max(self.bit_rates))  # 400
        # print("Max",max_bit_rate)
        # print("Current bitrate", current_bit_rate)
        
        # Optical_rl_gym style normalization
        # 100 → -1.0, 250 → 0.0, 400 → 1.0
        # if max_bit_rate > min_bit_rate:
        #     bit_rate_norm = 2.0 * (current_bit_rate - 0.5 * (min_bit_rate + max_bit_rate)) / (max_bit_rate - min_bit_rate)
        # else:
        #     bit_rate_norm = 0.0

        bit_rate_norm = current_bit_rate/max_bit_rate
        
        # ===================================================================
        # COMPONENT 1: Source and Destination One-Hot (2×|V|)
        # ===================================================================
        num_nodes = self.topology.number_of_nodes()
        source_one_hot = np.zeros(num_nodes, dtype=np.float32)
        dest_one_hot = np.zeros(num_nodes, dtype=np.float32)
        source_one_hot[self.topology.nodes[src]['index']] = 1.0
        dest_one_hot[self.topology.nodes[dst]['index']] = 1.0
        
        # ===================================================================
        # COMPONENT 2: Path-Specific Features (k × 7)
        # ===================================================================
        path_features = np.zeros(self.k_paths * 7, dtype=np.float32)
        
        for path_idx, path in enumerate(paths):
            path_length_km = path.length
            num_links = len(path.node_list) - 1
            num_spans = path.num_spans
            avg_link_length_km = path_length_km / num_links if num_links > 0 else 0.0
            
            # Get required slots (band-independent, use C-band as reference)
            omega_r_k = self.get_number_slots(path, self.num_bands, 0, self.modulations)
            
            # Get modulation format
            modulation_info = self.get_modulation_format(path, self.num_bands, 0, self.modulations)
            modulation_encoded = self.modulation_encoding.get(modulation_info['modulation'], 0.0)
            
            # Feature 1: Required slots
            # Expected range: [2, 9] for bit_rates=[100, 200, 400]
            # 100 Gbps: 2-3 slots, 200 Gbps: 3-5 slots, 400 Gbps: 5-9 slots (depending on modulation)
            min_slots = 2.0
            max_slots = 5.0
            if max_slots > min_slots:
                omega_norm = 2.0 * (omega_r_k - 0.5 * (min_slots + max_slots)) / (max_slots - min_slots)
            else:
                omega_norm = 0.0
            
            # Feature 2: Path length [min, max]
            if self.max_path_length > self.min_path_length:
                length_norm = 2.0 * (path_length_km - 0.5 * (self.min_path_length + self.max_path_length)) / (self.max_path_length - self.min_path_length)
            else:
                length_norm = 0.0

            
            # Feature 3: Num links [1, max]
            if self.max_num_links > 1:
                links_norm = 2.0 * (num_links - 0.5 * (1 + self.max_num_links)) / (self.max_num_links - 1)
            else:
                links_norm = 0.0
            
            # Feature 4: Num spans [min, max]
            if self.max_num_spans > self.min_num_spans:
                spans_norm = 2.0 * (num_spans - 0.5 * (self.min_num_spans + self.max_num_spans)) / (self.max_num_spans - self.min_num_spans)
            else:
                spans_norm = 0.0
            
            # Feature 5: Avg link length [min, max]
            if self.max_avg_link_length > self.min_avg_link_length:
                avg_link_norm = 2.0 * (avg_link_length_km - 0.5 * (self.min_avg_link_length + self.max_avg_link_length)) / (self.max_avg_link_length - self.min_avg_link_length)
            else:
                avg_link_norm = 0.0
            
            # Feature 6: Modulation [0.25, 1.0]
            # PM_BPSK: 0.25, PM_QPSK: 0.5, PM_8QAM: 0.75, PM_16QAM: 1.0
            modulation_norm = 2.0 * (modulation_encoded - 0.5 * (0.25 + 1.0)) / (1.0 - 0.25)
            
            # Feature 7: Placeholder
            placeholder = 0.0
            
            # Store path features
            base_idx = path_idx * 7
            path_features[base_idx:base_idx+7] = [
                omega_norm,
                length_norm,
                links_norm,
                spans_norm,
                avg_link_norm,
                modulation_norm,
                placeholder
            ]
        
        # ===================================================================
        # COMPONENT 3: Band-Specific Features (k × B × (2j+6))
        # ===================================================================
        features_per_path_band = 2 * self.j + 4
        band_features = np.zeros(self.k_paths * self.num_bands * features_per_path_band, dtype=np.float32)
        
        for path_idx, path in enumerate(paths):
            omega_r_k = self.get_number_slots(path, self.num_bands, 0, self.modulations)
            
            for band in range(self.num_bands):
                # Get band capacity and statistics
                band_start, band_end = self.get_shift(band)
                band_capacity = band_end - band_start
                #band_stats = self.band_max_stats[band]
                
                # Get available slots for this path-band
                available_slots = self.get_available_slots(path, band)
                
                # Feature 1: Total free slots [0, band_capacity]
                total_free = np.sum(available_slots)
                free_norm = 2.0 * (total_free - 0.5 * band_capacity) / band_capacity
                
                # Get all blocks
                initial_indices, values, lengths = RMSAEnv.rle(available_slots)
                available_indices = np.where(values == 1)[0]
                
                if len(available_indices) > 0:
                    all_block_sizes = lengths[available_indices]
                    all_block_starts = initial_indices[available_indices]
                    
                    # Feature 2: Total blocks [0, max_total_blocks]
                    total_blocks = len(all_block_sizes)
                    blocks_norm = 2.0 * (total_blocks - 0.5 * 15) / 15 # assuming 15 total blocks
                    
                    # Feature 3: Avg block size [min, max]
                    avg_size = np.mean(all_block_sizes)
                    min_avg = 12 #band_stats['min_avg_block_size'] 
                    max_avg = 4 # band_stats['max_avg_block_size']
                    if max_avg > min_avg:
                        avg_norm = 2.0 * (avg_size - 0.5 * (min_avg + max_avg)) / (max_avg - min_avg)
                    else:
                        avg_norm = 0.0
                    
                    # Filter satisfying blocks
                    satisfying_mask = all_block_sizes >= omega_r_k
                    satisfying_sizes = all_block_sizes[satisfying_mask]
                    satisfying_starts = all_block_starts[satisfying_mask]
                    
                    if len(satisfying_sizes) > 0:
                        # Feature 4: Num satisfying blocks [0, max_satisfying_blocks]
                        num_satisfying = len(satisfying_sizes)
                        sat_norm = 2.0 * (num_satisfying - 0.5 * 10) / 10 # assuming max 10 satisfying  blocks
                        
                        # Features 5 & 6: Block sizes and starts (first j)
                        block_sizes_norm = np.full(self.j, -1.0, dtype=np.float32)  # Padding
                        block_starts_norm = np.full(self.j, -1.0, dtype=np.float32)
                        
                        for i in range(min(self.j, len(satisfying_sizes))):
                            # Block size normalization [omega_r_k, band_capacity]
                            if band_capacity > omega_r_k:
                                block_sizes_norm[i] = 2.0 * (satisfying_sizes[i] - 0.5 * (omega_r_k + band_capacity)) / (band_capacity - omega_r_k)
                            else:
                                block_sizes_norm[i] = 0.0
                            
                            # Block start normalization [0, band_capacity - omega_r_k]
                            max_start = band_capacity - omega_r_k
                            if max_start > 0:
                                block_starts_norm[i] = 2.0 * (satisfying_starts[i] - 0.5 * max_start) / max_start
                            else:
                                block_starts_norm[i] = 0.0
                    else:
                        # No satisfying blocks
                        sat_norm = -1.0
                        block_sizes_norm = np.full(self.j, -1.0, dtype=np.float32)
                        block_starts_norm = np.full(self.j, -1.0, dtype=np.float32)
                else:
                    # No blocks available
                    blocks_norm = -1.0
                    avg_norm = -1.0
                    sat_norm = -1.0
                    block_sizes_norm = np.full(self.j, -1.0, dtype=np.float32)
                    block_starts_norm = np.full(self.j, -1.0, dtype=np.float32)
                
                # Combine band features
                band_feature_vector = np.concatenate([
                    [free_norm],           # 1
                    [blocks_norm],         # 1
                    [avg_norm],            # 1
                    [sat_norm],            # 1
                    block_sizes_norm,      # j
                    block_starts_norm      # j
                ])  # Total: 4 + 2j
                
                # Store in band_features array
                base_idx = (path_idx * self.num_bands + band) * features_per_path_band
                band_features[base_idx:base_idx+features_per_path_band] = band_feature_vector
        
        # ===================================================================
        # Combine All Components and Reshape
        # ===================================================================
        observation = np.concatenate([
            [bit_rate_norm],    # 1
            source_one_hot,     # |V|
            dest_one_hot,       # |V|
            path_features,      # k × 7
            band_features       # k × B × (2j+6)
        ]).astype(np.float32)
        
        # Clip to reasonable bounds (allow margin beyond [-1, 1])
        #observation = np.clip(observation, -2.0, 2.0)
        
        # Reshape to match observation space
        observation = observation.reshape(self.observation_space.shape)
        
        self.steps += 1
        
        return observation


    def step(self, action: int):
        """
        Execute one step in the environment by taking an action.
        
        The action is decoded into (path, band, block) selection, then validated
        and executed by the parent RMSA environment. Additional information about
        normalization state is added to the info dictionary.
        
        Args:
            action (int): Discrete action index to execute
            
        Returns:
            tuple: (observation, reward, done, info)
                - observation (np.array): New state after action
                - reward (float): Reward for this transition
                - done (bool): Whether episode is complete
                - info (dict): Additional diagnostic information
        """
        parent_step_result = None
        valid_action = False
        slots = -1  # Track slots required (for logging)

        # Check if action is a valid allocation (not rejection)
        if action < self.k_paths * self.j * self.num_bands:
            valid_action = True
            
            # Decode action into path, band, and block indices
            route, band, block = self._get_route_block_id(action)

            # Get available blocks for this path-band combination
            initial_indices, lengths = self.get_available_blocks(
                route, self.num_bands, band, self.modulations
            )
            
            # Get required slots for this path-band
            slots = self.get_number_slots(
                self.k_shortest_paths[
                    self.current_service.source, 
                    self.current_service.destination
                ][route], 
                self.num_bands, 
                band, 
                self.modulations
            )
            
            # Check if the requested block exists
            if block < len(initial_indices):
                # Valid block selection - execute allocation
                parent_step_result = super().step([route, band, initial_indices[block]])
            else:
                # Invalid block (doesn't exist) - treat as rejection
                parent_step_result = super().step(
                    [self.k_paths, self.num_bands, self.num_spectrum_resources]
                )
        else:
            # Explicit rejection action
            parent_step_result = super().step(
                [self.k_paths, self.num_bands, self.num_spectrum_resources]
            )

        # Unpack results from parent
        obs, rw, done, info = parent_step_result
        
        # Add DeepRMSA-specific information
        info['slots'] = slots
        info['normalization_enabled'] = self.normalization_enabled
        info['observation_steps'] = self.steps
        
        return parent_step_result

    def reset(self, only_episode_counters=True):
        """
        Reset the environment to initial state.
        
        Args:
            only_episode_counters (bool): If True, only reset episode-specific
                counters (soft reset between episodes). If False, perform full
                reset including normalization statistics (hard reset).
        
        Returns:
            np.array: Initial observation after reset
        """
        # Full reset: clear normalization statistics
        if not only_episode_counters:
            self.steps = 0
            self.normalization_enabled = True
            
            # Optional: Reset running statistics for fresh start
            # Uncomment to clear learned statistics between full resets
            # for normalizer in self.normalizers.values():
            #     normalizer.__init__()
        
        # Call parent reset
        return super().reset(only_episode_counters=only_episode_counters)

    def _get_route_block_id(self, action: int) -> Tuple[int, int, int]:
        """
        Decode a single action index into (route, band, block) indices.
        
        Action Encoding:
            action = route + (band × k_paths) + (block × k_paths × num_bands)
        
        Args:
            action (int): Discrete action index
            
        Returns:
            tuple: (route, band, block) indices
            
        Raises:
            ValueError: If decoded block index exceeds j-1
        """
        route = action % self.k_paths
        band = (action // self.k_paths) % self.num_bands
        block = action // (self.k_paths * self.num_bands)
        
        # FIX: Validate that block is within valid range
        if block >= self.j:
            raise ValueError(
                f"Invalid action {action}: decoded block index {block} exceeds "
                f"maximum allowed block index {self.j - 1}. "
                f"Action space size may be incorrectly configured."
            )
        
        return route, band, block

 



