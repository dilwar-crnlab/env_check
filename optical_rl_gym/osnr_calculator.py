"""
osnr_calculator.py

OSNR (Optical Signal-to-Noise Ratio) calculator for optical network simulation.
Implements the GN-ISRS model for calculating ASE noise and nonlinear interference.

Based on:
"ISRS impact-reduced routing, modulation, band, and spectrum allocation algorithm 
in C + L-bands elastic optical networks"

Author: Dilwar Barbhuiya
"""

from math import exp, pi, atan, asinh, log10
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

from optical_rl_gym.utils import PhysicalParameters

# Optional: Import logging utilities
try:
    from optical_rl_gym.logging_utils import OSNRCalculationLogger
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    OSNRCalculationLogger = None


class OSNRCalculator:
    """
    Calculate OSNR for optical services using the GN-ISRS model.
    
    The calculator computes:
    1. ASE (Amplified Spontaneous Emission) noise
    2. NLI (Nonlinear Interference) noise:
       - SCI (Self-Channel Interference)
       - XCI (Cross-Channel Interference)
    
    Features:
    - Multi-band support (C-band, L-band)
    - ISRS (Inter-band Stimulated Raman Scattering) effects
    - Band-filtered interference calculation
    - Optional detailed logging
    """
    
    def __init__(self, logger=None):
        """
        Initialize OSNR calculator.
        
        Args:
            logger: Optional ServiceRequestLogger for detailed logging
        """
        self.physical_params = PhysicalParameters()
        
        # Initialize logger if available
        if LOGGING_AVAILABLE and logger is not None:
            self.logger = OSNRCalculationLogger(parent_logger=logger)
        else:
            self.logger = None
    
    def calculate_dispersion_params(self, freq_r: float, freq_r_prime: float = None) -> float:
        """
        Calculate dispersion parameters φ_r or φ_{r,r'}.
        
        Implements equation (6) from the paper:
        - φ_r = β_2 + 2π·β_3·f_r
        - φ_{r,r'} = [β_2 + π·β_3·(f_r + f_r')] · (f_r - f_r')
        
        Args:
            freq_r: Center frequency of service r (Hz)
            freq_r_prime: Center frequency of service r' (Hz), optional
        
        Returns:
            float: Dispersion parameter φ_r or φ_{r,r'}
        
        Note:
            Returns small non-zero value for edge cases to avoid division by zero.
        """
        
        if freq_r_prime is None:
            # Single service dispersion: φ_r = β_2 + 2π·β_3·f_r
            dispersion_param = (self.physical_params.beta_2 + 
                               2 * pi * self.physical_params.beta_3 * freq_r)
            
            # Validate: Should not be exactly zero for any realistic frequency
            if abs(dispersion_param) < 1e-35:
                # Return small non-zero value to avoid division by zero
                return 1e-35 if dispersion_param >= 0 else -1e-35
        
        else:
            # Cross-service dispersion: φ_{r,r'} = [β_2 + π·β_3·(f_r + f_r')] · (f_r - f_r')
            
            # Check for same frequency (should be handled by caller, but be safe)
            freq_diff = freq_r - freq_r_prime  # FIXED: Correct sign as per equation (6)
            
            if abs(freq_diff) < 1e6:  # Less than 1 MHz apart
                # Services at essentially same frequency - no walk-off
                # Return small non-zero value to avoid division by zero
                return 1e-35
            
            bracket_term = (self.physical_params.beta_2 + 
                           pi * self.physical_params.beta_3 * (freq_r + freq_r_prime))
            
            dispersion_param = bracket_term * freq_diff
            
            # Return small non-zero if result is too small
            if abs(dispersion_param) < 1e-35:
                return 1e-35 if dispersion_param >= 0 else -1e-35
        
        return dispersion_param
    
    def calculate_ase_noise(self, service, topology) -> float:
        """
        Calculate ASE (Amplified Spontaneous Emission) noise power.
        
        Implements equation (2) from the paper:
        P_ASE = Σ_{l∈path} Σ_{s=1}^{N_l} 2·n_sp·h·f_r·B_r·(e^{αL_s} - 1)
        
        Args:
            service: Service object containing path, center_frequency, bandwidth
            topology: Network topology with link and span information
            
        Returns:
            float: ASE noise power in Watts
        """
        p_ase = 0.0
        path = service.path
        
        # Constant term: 2 * n_sp * h * f_r * B_r
        constant_term = (2 * self.physical_params.nse * 
                        self.physical_params.h_plank * 
                        service.center_frequency * 
                        service.bandwidth)
        
        # Sum over all links in the path
        for link in path.links:
            # Sum over all spans in each link
            for span in link.spans:
                # Calculate ASE contribution per span: (e^{α*L} - 1)
                span_contribution = (exp(self.physical_params.alpha * span.length) - 1)
                p_ase += constant_term * span_contribution
        
        return p_ase
    
    def calculate_sci(self, service, current_node, next_node, topology):
        """
        Calculate SCI (Self-Channel Interference).
        
        Implements equation (4) from the paper.
        Includes ISRS (Inter-band Stimulated Raman Scattering) effects.
        
        Args:
            service: Service object
            current_node: Current node in the directed path
            next_node: Next node in the directed path
            topology: Network topology
        
        Returns:
            float: SCI noise power in Watts
        """
        link = topology[current_node][next_node]['link']
        n_spans = len(link.spans)
        
        # Get ALL services on this link for ISRS calculation
        all_services = topology[current_node][next_node]['running_services']
        d_l = len(all_services)  # All services contribute to ISRS power transfer
        
        # Calculate dispersion parameter
        phi_r = self.calculate_dispersion_params(service.center_frequency)
        
        # Safety check: Avoid division by zero
        denominator = phi_r * service.bandwidth**2
        if abs(denominator) < 1e-40:
            # Dispersion too small - SCI calculation would be unstable
            return 0.0
        
        # Main factor: N_l * (8/81) * (γ²P³) / (πα²) * 1/(φ_r·B_r²)
        factor = (n_spans * (8/81) * 
                 (self.physical_params.gamma**2 * self.physical_params.launch_power**3) / 
                 (pi * self.physical_params.alpha**2) * 
                 (1 / denominator))
        
        # ISRS term: (2α - D_l·P·C_k·f_r)²
        # Note: D_l counts ALL services (all bands) for ISRS power transfer
        isrs_term = (2 * self.physical_params.alpha - 
                    d_l * self.physical_params.launch_power * 
                    self.physical_params.cr * service.center_frequency)**2
        
        # Calculate terms for asinh functions
        term1 = (isrs_term - self.physical_params.alpha**2) / self.physical_params.alpha
        term2 = (4 * self.physical_params.alpha**2 - isrs_term) / (2 * self.physical_params.alpha)
        
        # Calculate SCI
        p_sci = factor * (
            term1 * asinh((3 * pi / (2 * self.physical_params.alpha)) * phi_r * service.bandwidth**2) +
            term2 * asinh((3 * pi / (4 * self.physical_params.alpha)) * phi_r * service.bandwidth**2)
        )
        
        return p_sci
    
    def calculate_xci(self, service, current_node: str, next_node: str, topology) -> float:
        """
        Calculate XCI (Cross-Channel Interference).
        
        Implements equation (5) from the paper.
        
        CRITICAL FIX: Only considers services in the SAME BAND for XCI calculation.
        - ISRS (power transfer): Inter-band effect, uses all services
        - XCI (four-wave mixing): Intra-band effect, uses same-band services only
        
        Args:
            service: Current service for which XCI is calculated
            current_node: Current node in the directed path
            next_node: Next node in the directed path  
            topology: Network topology with link and running services information
            
        Returns:
            float: XCI noise power in Watts
        """
        link = topology[current_node][next_node]['link']
        n_spans = len(link.spans)
        
        # Get ALL services for ISRS calculation (power transfer)
        all_services = topology[current_node][next_node]['running_services']
        d_l = len(all_services)  # All services contribute to ISRS
        
        # CRITICAL FIX: Filter to SAME BAND for XCI calculation
        # Four-wave mixing only occurs between services in the same band
        same_band_services = [s for s in all_services 
                             if hasattr(s, 'band') and s.band == service.band]
        
        # If no other services in the same band on this link, XCI = 0
        if len(same_band_services) <= 1:
            return 0.0
        
        # Main factor: N_l * (16/81) * (γ²P³) / (π²α²)
        main_factor = (n_spans * (16/81) * 
                      (self.physical_params.gamma**2 * self.physical_params.launch_power**3) / 
                      (pi**2 * self.physical_params.alpha**2))
        
        p_xci = 0.0
        
        # Sum over all interfering services in the SAME BAND (r' ≠ r)
        for other_service in same_band_services:
            if other_service.service_id == service.service_id:
                continue  # Skip self-interference
            
            # Calculate cross-dispersion parameter φ_{r,r'}
            phi_r_r_prime = self.calculate_dispersion_params(
                service.center_frequency, 
                other_service.center_frequency
            )
            
            # Safety check: Skip if denominator too small
            denominator = phi_r_r_prime * other_service.bandwidth
            if abs(denominator) < 1e-30:
                # Services too close in frequency or dispersion too small
                continue  # Skip this interferer
            
            # Calculate ISRS term for interfering service r'
            # Note: Uses d_l (ALL services) for ISRS effect
            isrs_term_r_prime = (2 * self.physical_params.alpha - 
                                d_l * self.physical_params.launch_power * 
                                self.physical_params.cr * other_service.center_frequency)**2
            
            # Calculate terms for interfering service r'
            term1_r_prime = ((isrs_term_r_prime - self.physical_params.alpha**2) / 
                            self.physical_params.alpha)
            
            term2_r_prime = ((4 * self.physical_params.alpha**2 - isrs_term_r_prime) / 
                            (2 * self.physical_params.alpha))
            
            # Calculate the inner sum factor: 1/(φ_{r,r'} * B_{r'})
            inner_sum_factor = 1 / denominator
            
            # Calculate atan terms with current service bandwidth B_r
            atan_term1 = atan((2 * pi**2 / self.physical_params.alpha) * 
                            phi_r_r_prime * service.bandwidth)
            
            atan_term2 = atan((pi**2 / self.physical_params.alpha) * 
                            phi_r_r_prime * service.bandwidth)
            
            # Sum contribution from this interfering service
            service_contribution = inner_sum_factor * (
                term1_r_prime * atan_term1 + 
                term2_r_prime * atan_term2
            )
            
            p_xci += service_contribution
        
        # Apply main factor
        p_xci *= main_factor
        
        return p_xci
    
    def calculate_osnr(self, service, topology) -> float:
        """
        Calculate OSNR (Optical Signal-to-Noise Ratio) for a service.
        
        Implements equation (1) from the paper:
        OSNR = P / (P_ASE + P_NLI)
        
        Where:
        - P: Launch power
        - P_ASE: ASE noise (equation 2)
        - P_NLI: Nonlinear interference = SCI + XCI (equations 3, 4, 5)
        
        Args:
            service: Service object with path, frequency, bandwidth, etc.
            topology: Network topology with running services
        
        Returns:
            float: OSNR in dB
        """
        
        # ====================================================================
        # LOGGING: Calculation Start
        # ====================================================================
        if self.logger:
            self.logger.log_calculation_start(service.service_id)
        
        # ====================================================================
        # Calculate ASE Noise
        # ====================================================================
        p_ase = self.calculate_ase_noise(service, topology)
        
        if self.logger:
            self.logger.log_ase(
                service.service_id, 
                p_ase,
                num_spans=sum(len(link.spans) for link in service.path.links),
                bandwidth=service.bandwidth
            )
        
        # ====================================================================
        # Calculate NLI Noise (Link by Link)
        # ====================================================================
        p_nli_total = 0
        path = service.path.node_list
        link_details = []
        
        # Iterate through directed path
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            
            # Calculate SCI for this directed edge
            p_sci = self.calculate_sci(service, current_node, next_node, topology)
            
            # Calculate XCI for this directed edge (same band only)
            p_xci = self.calculate_xci(service, current_node, next_node, topology)
            
            # Total NLI for this link
            p_nli_link = p_sci + p_xci
            p_nli_total += p_nli_link
            
            # Store details for logging
            if self.logger:
                link_details.append({
                    'link': f"{current_node}→{next_node}",
                    'link_index': topology[current_node][next_node]['index'],
                    'num_spans': len(topology[current_node][next_node]['link'].spans),
                    'num_services': len(topology[current_node][next_node]['running_services']),
                    'p_sci': p_sci,
                    'p_xci': p_xci,
                    'p_nli_link': p_nli_link
                })
        
        # ====================================================================
        # Calculate Final OSNR
        # ====================================================================
        # Handle edge case of zero/negative noise
        total_noise = p_ase + p_nli_total
        if total_noise <= 0:
            osnr_db = 100.0  # Very high OSNR (essentially no noise)
        else:
            osnr = self.physical_params.launch_power / total_noise
            osnr_db = 10 * log10(osnr) if osnr > 0 else -100.0
        
        # ====================================================================
        # LOGGING: Calculation Complete
        # ====================================================================
        if self.logger:
            self.logger.log_complete(
                service_id=service.service_id,
                p_ase=p_ase,
                p_nli_total=p_nli_total,
                osnr_linear=self.physical_params.launch_power / total_noise if total_noise > 0 else 1e10,
                osnr_db=osnr_db,
                link_details=link_details
            )
        
        return osnr_db


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def test_osnr_calculator():
    """
    Test OSNR calculator with sample parameters.
    Useful for validating calculations and debugging.
    """
    from optical_rl_gym.utils import Service, Path, Link, Span
    
    print("="*70)
    print("OSNR Calculator Test")
    print("="*70)
    
    # Create calculator
    calc = OSNRCalculator()
    
    # Test dispersion calculations
    print("\n1. Dispersion Parameter Tests:")
    print("-" * 70)
    
    # C-band frequencies
    freq_c_low = 191.7e12   # C-band start
    freq_c_high = 196.0e12  # C-band end
    freq_c_mid = 193.85e12  # C-band middle
    
    # L-band frequencies
    freq_l_low = 185.7e12   # L-band start
    freq_l_high = 191.7e12  # L-band end
    
    print("\nSingle Service Dispersion (φ_r):")
    for freq in [freq_c_low, freq_c_mid, freq_c_high, freq_l_low, freq_l_high]:
        phi = calc.calculate_dispersion_params(freq)
        print(f"  f = {freq/1e12:.2f} THz: φ_r = {phi:.3e}")
    
    print("\nCross-Service Dispersion (φ_{r,r'}):")
    
    # Same band (should be small)
    phi_same = calc.calculate_dispersion_params(freq_c_mid, freq_c_mid + 100e9)
    print(f"  Same band, 100 GHz apart: φ_{{r,r'}} = {phi_same:.3e}")
    
    # Different bands (should be larger)
    phi_diff = calc.calculate_dispersion_params(freq_c_mid, freq_l_low)
    print(f"  Different bands (C vs L): φ_{{r,r'}} = {phi_diff:.3e}")
    
    # Check for division safety
    print("\n2. Division Safety Check:")
    print("-" * 70)
    bandwidth = 62.5e9  # 62.5 GHz
    denominator = phi_same * bandwidth
    print(f"  φ_{{r,r'}} * B = {denominator:.3e}")
    print(f"  Safe for division: {abs(denominator) > 1e-30}")
    
    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)


if __name__ == "__main__":
    test_osnr_calculator()
