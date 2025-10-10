
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
import numpy as np

from .rmsa_env import RMSAEnv
from .optical_network_env import OpticalNetworkEnv


class RunningStats:
    """
    Computes running mean and standard deviation using Welford's online algorithm.
    This algorithm is numerically stable and memory efficient, suitable for 
    real-time statistics computation during RL training.
    
    Reference: Welford, B. P. (1962). "Note on a method for calculating 
    corrected sums of squares and products"
    """
    
    def __init__(self):
        """Initialize running statistics counters."""
        self.n = 0              # Number of samples seen
        self.mean = 0.0         # Running mean
        self.M2 = 0.0           # Sum of squared differences from mean
    
    def update(self, x):
        """
        Update statistics with a new value using Welford's algorithm.
        
        Args:
            x (float): New value to incorporate into statistics
        """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    def normalize(self, x):
        """
        Normalize a value using computed mean and standard deviation.
        
        Args:
            x (float): Value to normalize
            
        Returns:
            float: Normalized value (z-score)
        """
        if self.n < 2:
            # Not enough samples for reliable statistics
            return 0.0
        
        # Compute sample variance and standard deviation
        variance = self.M2 / (self.n - 1)
        std = np.sqrt(variance)
        
        if std < 1e-8:
            # Avoid division by zero for constant values
            return 0.0
        
        # Return z-score: (x - mean) / std
        return (x - self.mean) / std
    
    def get_stats(self):
        """
        Get current mean and standard deviation for debugging.
        
        Returns:
            tuple: (mean, std)
        """
        if self.n < 2:
            return self.mean, 0.0
        variance = self.M2 / (self.n - 1)
        return self.mean, np.sqrt(variance)


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
        j=5,
        episode_length=1000,
        mean_service_holding_time=25.0,
        mean_service_inter_arrival_time=0.1,
        node_request_probabilities=None,
        seed=None,
        bit_rates: Sequence = [50, 100, 200],
        k_paths=5,
        allow_rejection=False,
        use_statistical_normalization=False,
        warmup_steps=1000,
    ):
        """
        Initialize DeepRMSA environment.
        
        Args:
            num_bands (int): Number of spectral bands (1=C-band, 2=C+L bands)
            topology (nx.Graph): Network topology graph
            j (int): Number of available blocks to consider per path-band
            episode_length (int): Number of service requests per episode
            mean_service_holding_time (float): Mean holding time in seconds
            mean_service_inter_arrival_time (float): Mean inter-arrival time
            node_request_probabilities (np.array): Source-dest pair probabilities
            seed (int): Random seed for reproducibility
            k_paths (int): Number of candidate paths (K-shortest paths)
            allow_rejection (bool): Whether rejection is a valid action
            use_statistical_normalization (bool): Use running stats for normalization
            warmup_steps (int): Steps before enabling statistical normalization
        """
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
        
        # Normalization parameters
        self.use_statistical_normalization = use_statistical_normalization
        self.warmup_steps = warmup_steps
        self.steps = 0

        # FIX: Correct initialization logic
        if self.use_statistical_normalization:
            # Statistical mode: Start disabled (use domain-based during warmup)
            self.normalization_enabled = False
        else:
            # Domain-based mode: Always enabled
            self.normalization_enabled = True

        
        # Compute longest path statistics for domain-based normalization
        # These serve as upper bounds during warmup period
        self._compute_longest_path_stats()
        
        # Initialize running statistics for each feature type
        # During warmup, we collect statistics; after warmup, we normalize
        self.normalizers = {
            'block_idx': RunningStats(),           # Starting index of blocks
            'block_size': RunningStats(),          # Size of blocks
            'required_fss': RunningStats(),        # Required frequency slots
            'avg_block_size': RunningStats(),      # Average block size
            'total_available': RunningStats(),     # Total available slots
            'fragmentation': RunningStats(),       # RMS fragmentation metric
            'path_length': RunningStats(),         # Path length in km
            'num_links': RunningStats(),           # Number of links in path
            'num_spans': RunningStats(),           # Number of spans in path
            'avg_link_length': RunningStats(),     # Average link length
            'avg_osnr_margin_c': RunningStats(),   # C-band OSNR margin
            'avg_osnr_margin_l': RunningStats(),   # L-band OSNR margin
        }
        
        # Modulation format encoding for observation
        # Higher values = more efficient modulation formats
        self.modulation_encoding = {
            'PM_BPSK': 0.25,   # Lowest spectral efficiency
            'PM_QPSK': 0.50,
            'PM_8QAM': 0.75,
            'PM_16QAM': 1.0    # Highest spectral efficiency
        }
        
       # Calculate observation space dimensions
        num_nodes = self.topology.number_of_nodes()
        path_features_per_path = 7                    # Required slots, length, links, spans, avg length, modulation, placeholder
        band_features_per_path_band = 2 * self.j + 5 # j indices + j sizes + 5 aggregate features

        observation_size = (
            2 * num_nodes +                                           # Source + dest
            self.k_paths * path_features_per_path +                   # Path features
            self.k_paths * self.num_bands * band_features_per_path_band  # Band features
        )

        
        # # Print configuration summary
        # print(f"\n{'='*60}")
        # print(f"{'DeepRMSA Environment Configuration':^60}")
        # print(f"{'='*60}")
        # print(f"Topology nodes: {num_nodes}")
        # print(f"Number of bands: {self.num_bands} ({'C+L' if self.num_bands == 2 else 'C-only'})")
        # print(f"Candidate paths (K): {self.k_paths}")
        # print(f"Blocks per path-band (j): {self.j}")
        # print(f"\nObservation Space:")
        # print(f"  Total size: {observation_size}")
        # print(f"\nObservation Space:")
        # print(f"  Total size: {observation_size}")
        # print(f"  - Source/Dest one-hot: {2*num_nodes}")
        # print(f"  - Path features: {self.k_paths} × {path_features_per_path} = {self.k_paths * path_features_per_path}")
        # print(f"  - Band features: {self.k_paths} × {self.num_bands} × {band_features_per_path_band} = {self.k_paths * self.num_bands * band_features_per_path_band}")
        
        
        # Define observation space with wider bounds for standardized features
        self.observation_space = gym.spaces.Box(
            low=-2**30,   
            high=2**30,
            shape=(observation_size,), 
            dtype=np.float32
        )


        # Define action space: (path, band, block) combinations + rejection
        # Action encoding: action = route + (band × k_paths) + (block × k_paths × num_bands)
        self.action_space = gym.spaces.Discrete(
            self.k_paths * self.num_bands * self.j + self.reject_action
        )
        
        print(f"\nAction Space:")
        print(f"  Total size: {self.action_space.n}")
        print(f"  - Valid allocations: {self.k_paths * self.num_bands * self.j}")
        print(f"    ({self.k_paths} paths × {self.num_bands} bands × {self.j} blocks)")
        print(f"  - Rejection actions: {self.reject_action}")
        
        print(f"\nNormalization:")
        print(f"  Method: {'Statistical (z-score)' if self.use_statistical_normalization else 'Domain-based'}")
        print(f"  Warmup steps: {self.warmup_steps}")
        print(f"{'='*60}\n")

        # Seed spaces for reproducibility
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        # Perform initial reset
        self.reset(only_episode_counters=False)

    def get_action_mask(self):
        """Generate action mask for current state"""
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        
        src = self.current_service.source
        dst = self.current_service.destination
        paths = self.k_shortest_paths[(src, dst)]
        
        for path_idx in range(self.k_paths):
            for band in range(self.num_bands):
                initial_indices, lengths = self.get_available_blocks(
                    path_idx, self.num_bands, band, self.modulations
                )
                
                required_slots = self.get_number_slots(
                    paths[path_idx], self.num_bands, band, self.modulations
                )
                
                for block_idx in range(min(self.j, len(initial_indices))):
                    if lengths[block_idx] >= required_slots:
                        action = path_idx + (band * self.k_paths) + \
                                (block_idx * self.k_paths * self.num_bands)
                        mask[action] = 1.0
        
        if self.reject_action > 0:
            mask[-1] = 1.0
        
        return mask

    def _compute_longest_path_stats(self):
        """
        Compute statistics of the longest path across all K-shortest paths.
        
        These statistics are used for domain-based normalization during the 
        warmup period before statistical normalization is enabled. They provide
        reasonable upper bounds for normalizing path-related features.
        
        Computes and stores:
            - max_path_length: Longest path length in km
            - max_num_links: Maximum number of links in any path
            - max_num_spans: Maximum number of amplifier spans
            - max_avg_link_length: Maximum average link length
        """
        max_path_length = 0.0
        max_num_links = 0
        max_num_spans = 0
        max_avg_link_length = 0.0
        longest_path = None
        
        # Iterate through all source-destination pairs and their K paths
        for src_dst_pair, paths in self.k_shortest_paths.items():
            for path in paths:
                # Convert path length from meters to kilometers
                path_length_km = path.length
                
                # Count links (edges between consecutive nodes)
                num_links = len(path.node_list) - 1
                
                # Count total spans across all links
                num_spans = self._calculate_num_spans(path)
                
                # Calculate average link length
                avg_link_length = path_length_km / num_links if num_links > 0 else 0.0
                
                # Track maximum values
                if path_length_km > max_path_length:
                    max_path_length = path_length_km
                    longest_path = path
                
                max_num_links = max(max_num_links, num_links)
                max_num_spans = max(max_num_spans, num_spans)
                max_avg_link_length = max(max_avg_link_length, avg_link_length)
        
        # Store for use in normalization
        self.max_path_length = max_path_length
        self.max_num_links = max_num_links
        self.max_num_spans = max_num_spans
        self.max_avg_link_length = max_avg_link_length
        
        # Print summary for verification
        print(f"\n{'='*60}")
        print(f"{'Longest Path Statistics (Normalization Bounds)':^60}")
        print(f"{'='*60}")
        if longest_path:
            print(f"Longest path: {' → '.join(longest_path.node_list)}")
            print(f"  Length: {self.max_path_length:.2f} km")
            print(f"  Links: {self.max_num_links}")
            print(f"  Spans: {self.max_num_spans}")
            print(f"  Avg link length: {self.max_avg_link_length:.2f} km")
        else:
            print("No paths found in topology")
        print(f"{'='*60}\n")

    def _calculate_num_spans(self, path):
        """
        Calculate the total number of amplifier spans in a path.
        
        Each link in the optical network is divided into spans, with optical
        amplifiers placed at span boundaries. The span information is directly
        available in the path object structure.
        
        Args:
            path (Path): Path object containing link and span information
            
        Returns:
            int: Total number of spans across all links in the path
        """
        total_spans = 0
        
        # Path.links is a tuple of Link objects
        # Each Link has a .spans attribute which is a tuple of Span objects
        if hasattr(path, 'links') and path.links:
            for link in path.links:
                if hasattr(link, 'spans') and link.spans:
                    total_spans += len(link.spans)
        
        return total_spans

    def _calculate_path_band_osnr_margin(self, path, band):
        """
        Calculate average OSNR margin of active lightpaths on this path in this band.
        
        This provides a measure of how "congested" a particular path-band combination
        is in terms of optical signal quality. Lower margins indicate the path is
        heavily used and new allocations may cause interference.
        
        Args:
            path (Path): Path object to analyze
            band (int): Band ID (0 for C-band, 1 for L-band)
        
        Returns:
            float: Average OSNR margin (dB) of active services sharing links 
                   with this path in this band. Returns 0.0 if no services found.
        """
        # Extract links from this path as (node1, node2) tuples
        path_links = [(path.node_list[i], path.node_list[i+1]) 
                      for i in range(len(path.node_list)-1)]
        
        # Get currently running services from the network
        running_services = self.topology.graph.get("running_services", [])
        
        # Collect OSNR margins from relevant services
        osnr_margins = []
        
        for service in running_services:
            # Check if service is in the same band
            if hasattr(service, 'band') and service.band == band:
                # Check if service has path information
                if hasattr(service, 'path') and service.path:
                    # Extract service path links
                    service_links = [(service.path.node_list[i], service.path.node_list[i+1])
                                   for i in range(len(service.path.node_list)-1)]
                    
                    # Check if paths share any links (potential interference)
                    if set(path_links).intersection(set(service_links)):
                        # Include this service's OSNR margin
                        if hasattr(service, 'OSNR_margin'):
                            osnr_margins.append(service.OSNR_margin)
        
        # Return average margin, or 0 if no relevant services
        return np.mean(osnr_margins) if len(osnr_margins) > 0 else 0.0

    def observation(self):
        """
        Observation Structure:
            s_t = [source_onehot | dest_onehot | path_features | band_features]
            
        Path features (k paths × 7 features):
            - Required slots (absolute, normalized by max possible ~20)
            - Path length, num links, num spans, avg link length
            - Highest modulation, (placeholder for future path-specific features)
            
        Band features (k paths × B bands × (2j+5) features):
            - j block indices, j block sizes
            - Avg block size, total available FS, fragmentation
            - Required slots normalized by band capacity
            - OSNR margin
        """
        src = self.current_service.source
        dst = self.current_service.destination

        # Pre-compute band capacities
        band_capacities = {}
        for band in range(self.num_bands):
            band_start, band_end = self.get_shift(band)
            band_capacities[band] = band_end - band_start
        
        # ===================================================================
        # COMPONENT 1: Source and Destination One-Hot (2×|V|)
        # ===================================================================
        num_nodes = self.topology.number_of_nodes()
        source_one_hot = np.zeros(num_nodes)
        dest_one_hot = np.zeros(num_nodes)
        source_one_hot[self.topology.nodes[src]['index']] = 1
        dest_one_hot[self.topology.nodes[dst]['index']] = 1
        
        # ===================================================================
        # COMPONENT 2: Path-Specific Features (k × 7)
        # ===================================================================
        paths = self.k_shortest_paths[(src, dst)]
        path_features = []
        
        for path_idx, path in enumerate(paths):
            path_length_km = path.length
            num_links = len(path.node_list) - 1
           # num_spans = self._calculate_num_spans(path)
            num_spans = path.num_spans
            avg_link_length_km = path_length_km / num_links if num_links > 0 else 0.0
            
            # Get required slots using C-band as reference (band-independent metric)
            omega_r_k = self.get_number_slots(path, self.num_bands, 0, self.modulations)
            
            # Get modulation (path-length dependent only)
            modulation_info = self.get_modulation_format(path, self.num_bands, 0, self.modulations)
            modulation_encoded = self.modulation_encoding.get(modulation_info['modulation'], 0.0)
            
            # Normalize path features
            required_slots_norm = omega_r_k / 20.0 if omega_r_k > 0 else 0.0  # Fixed normalization
            path_length_norm = path_length_km / self.max_path_length if self.max_path_length > 0 else 0.0
            num_links_norm = num_links / self.max_num_links if self.max_num_links > 0 else 0.0
            num_spans_norm = num_spans / self.max_num_spans if self.max_num_spans > 0 else 0.0
            avg_link_length_norm = avg_link_length_km / self.max_avg_link_length if self.max_avg_link_length > 0 else 0.0
            
            # 7 path-specific features
            path_features.extend([
                required_slots_norm,      # 1
                path_length_norm,         # 2
                num_links_norm,           # 3
                num_spans_norm,           # 4
                avg_link_length_norm,     # 5
                modulation_encoded,       # 6
                0.0                       # 7 - placeholder for future features
            ])
        
        # ===================================================================
        # COMPONENT 3: Band-Specific Features (k × B × (2j+5))
        # ===================================================================
        band_features = []
        
        for path_idx, path in enumerate(paths):
            # Get required slots for this path (same absolute value for all bands)
            omega_r_k = self.get_number_slots(path, self.num_bands, 0, self.modulations)
            
            for band in range(self.num_bands):
                band_capacity = band_capacities[band]
                
                # Get available blocks for this path-band
                initial_indices, lengths = self.get_available_blocks(
                    path_idx, self.num_bands, band, self.modulations
                )
                
                # Block indices (j features)
                # Block indices (j features) - 1-indexed to avoid conflict with padding
                block_indices = []
                for block_idx in range(self.j):
                    if block_idx < len(initial_indices):
                        # Add 1 to distinguish real slot 0 from padding (0.0)
                        norm_idx = (initial_indices[block_idx] + 1) / (band_capacity + 1)
                        block_indices.append(norm_idx)
                    else:
                        block_indices.append(0.0)  # Only padding is exactly 0.0
                
                # Block sizes (j features)
                block_sizes = []
                for block_idx in range(self.j):
                    if block_idx < len(lengths):
                        norm_size = lengths[block_idx] / band_capacity
                        block_sizes.append(norm_size)
                    else:
                        block_sizes.append(0.0)
                
                # Average block size
                if len(lengths) > 0:
                    avg_block_size = np.mean(lengths) / band_capacity
                else:
                    avg_block_size = 0.0
                
                # Total available slots
                if len(lengths) > 0:
                    total_available_fss = np.sum(lengths) / band_capacity
                else:
                    total_available_fss = 0.0
                
                # Fragmentation
                try:
                    rms_frag = self.calculate_path_rms(path_idx, band)
                    fragmentation = min(1.0, rms_frag / 50.0) if not np.isinf(rms_frag) and rms_frag > 0 else 0.0
                except Exception:
                    fragmentation = 0.0
                
                # Required slots normalized by THIS band's capacity
                required_fss_band_norm = omega_r_k / band_capacity if omega_r_k > 0 else 0.0
                
                # OSNR margin
                path_band_osnr_margin = self._calculate_path_band_osnr_margin(path, band)
                osnr_margin_norm = min(1.0, path_band_osnr_margin / 10.0) if path_band_osnr_margin > 0 else 0.0
                
                # Combine band-specific features (2j + 5 features)
                band_features.extend(block_indices)           # j features
                band_features.extend(block_sizes)             # j features
                band_features.append(avg_block_size)          # 1
                band_features.append(total_available_fss)     # 2
                band_features.append(fragmentation)           # 3
                band_features.append(required_fss_band_norm)  # 4 - band-specific normalization
                band_features.append(osnr_margin_norm)        # 5
        
        # ===================================================================
        # Combine All Components
        # ===================================================================
        observation = np.concatenate([
            source_one_hot,     # |V|
            dest_one_hot,       # |V|
            path_features,      # k × 7
            band_features       # k × B × (2j+5)
        ])
        
        observation = np.clip(observation, 0.0, 1.0)
        self.steps += 1
        
        return observation.astype(np.float32)


    def _print_normalization_stats(self):
        """
        Print current normalization statistics for debugging and verification.
        
        This method displays the mean and standard deviation computed for each
        feature type, which helps verify that normalization is working correctly
        and features are on similar scales.
        """
        print(f"\n{'='*70}")
        print(f"{'Normalization Statistics':^70}")
        print(f"{'='*70}")
        print(f"{'Feature':<25} {'Mean':>12} {'Std':>12} {'Samples':>10}")
        print(f"{'-'*70}")
        
        for name, normalizer in self.normalizers.items():
            mean, std = normalizer.get_stats()
            print(f"{name:<25} {mean:>12.3f} {std:>12.3f} {normalizer.n:>10}")
        
        print(f"{'='*70}\n")

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

    def calculate_path_rms(self, path_index: int, band: int):
        """
        Calculate RMS (Root Mean Square) fragmentation metric for a path-band.
        
        The RMS metric quantifies spectrum fragmentation using the formula:
            RMS = (β × |A|) / √(Σ(αw²) / |A|)
        
        Where:
            - β: Maximum used spectrum slot index
            - |A|: Number of available spectrum fragments (islands)
            - αw: Width of each fragment
        
        Higher RMS values indicate more fragmentation, making it harder to
        allocate contiguous spectrum blocks for new services.
        
        Args:
            path_index (int): Index of path in k_shortest_paths for current src-dst
            band (int): Band ID (0=C-band, 1=L-band)
            
        Returns:
            float: RMS fragmentation factor (0 = no fragmentation, inf = fully fragmented)
        """
        # Get the path object
        path = self.k_shortest_paths[
            self.current_service.source, 
            self.current_service.destination
        ][path_index]
        
        # Get available slots along entire path (intersection of all links)
        available_slots = self.get_available_slots(path, band)
        
        if len(available_slots) == 0:
            return 0.0
        
        # Find occupied slots
        occupied_slots = np.where(available_slots == 0)[0]
        if len(occupied_slots) == 0:
            # No fragmentation if no slots are occupied
            return 0.0
        
        # β: Maximum spectrum range considered
        #beta = len(occupied_slots)
        beta = np.max(occupied_slots) + 1  # +1 because slot indices start from 0
        
        
        # Use Run-Length Encoding to find contiguous blocks
        initial_indices, values, lengths = RMSAEnv.rle(available_slots)
        
        # Get indices where spectrum is available (value == 1)
        available_indices = np.where(values == 1)[0]
        
        if len(available_indices) == 0:
            # Maximum fragmentation - no available slots
            return float('inf')
        
        # Get widths of available fragments (spectrum islands)
        island_widths = lengths[available_indices]
        
        # Calculate RMS metric
        num_islands = len(island_widths)  # |A|
        sum_of_squares = np.sum(island_widths ** 2)  # Σ(αw²)
        
        if sum_of_squares == 0:
            return 0.0
        
        # RMS = (β × |A|) / √(Σ(αw²) / |A|)
        rms_denominator = np.sqrt(sum_of_squares / num_islands)
        rms_factor = (beta * num_islands) / rms_denominator
        
        return rms_factor




