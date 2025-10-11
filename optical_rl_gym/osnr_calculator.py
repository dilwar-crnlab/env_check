
'''
Author: Dilwar Barbhuiya
'''


from math import exp, pi, atan, asinh, log10
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from optical_rl_gym.utils import PhysicalParameters

class OSNRCalculator:
    def __init__(self):
        self.physical_params = PhysicalParameters()

    def calculate_dispersion_params(self, freq_r: float, freq_r_prime: float = None) -> float:
        """Calculate dispersion parameters phi_r or phi_r_r_prime"""
        if freq_r_prime is None:
            return self.physical_params.beta_2 + 2 * pi* self.physical_params.beta_3 * freq_r
        else:
            return (self.physical_params.beta_2 + 
                   pi*self.physical_params.beta_3*(freq_r + freq_r_prime))*(freq_r_prime - freq_r)



    def calculate_ase_noise(self, service, topology) -> float:
        """
        Calculate ASE noise power according to equation (2) from the paper.
        
        Args:
            service: Service object containing path, center_frequency, bandwidth
            topology: Network topology with link and span information
            
        Returns:
            float: ASE noise power in Watts
        """
        p_ase = 0.0
        path = service.path
        
        # Constant term: 2 * nsp * h * fr * Br
        constant_term = (2 * self.physical_params.nse * 
                        self.physical_params.h_plank * 
                        service.center_frequency * 
                        service.bandwidth)
        
        # Sum over all links in the path
        for link in path.links:
            # Sum over all spans in each link
            for span in link.spans:
                # Calculate ASE contribution per span: (e^(α*L) - 1)
                span_contribution = (exp(self.physical_params.alpha * span.length) - 1)
                p_ase += constant_term * span_contribution
        
        return p_ase

    def calculate_sci(self, service, current_node, next_node, topology):
        link = topology[current_node][next_node]['link']
        n_spans = len(link.spans)
        d_l = len(topology[current_node][next_node]['running_services'])
        
        # Calculate dispersion parameter
        phi_r = self.calculate_dispersion_params(service.center_frequency)
        
        # Main factor
        factor = n_spans * (8/81) * (self.physical_params.gamma**2 * 
                                    self.physical_params.launch_power**3) / \
                (pi * self.physical_params.alpha**2) * (1/(phi_r * service.bandwidth**2))
        
        # ISRS term
        isrs_term = (2*self.physical_params.alpha - 
                    d_l*self.physical_params.launch_power*
                    self.physical_params.cr*service.center_frequency)**2
        
        # Calculate terms
        term1 = (isrs_term - self.physical_params.alpha**2) / self.physical_params.alpha
        term2 = (4*self.physical_params.alpha**2 - isrs_term) / (2*self.physical_params.alpha)
        
        # Calculate SCI
        p_sci = factor * (
            term1 * asinh((3*pi/(2*self.physical_params.alpha)) * phi_r * service.bandwidth**2) +
            term2 * asinh((3*pi/(4*self.physical_params.alpha)) * phi_r * service.bandwidth**2)
        )
        return p_sci





    def calculate_xci(self, service, current_node: str, next_node: str, topology) -> float:
        """
        Calculate XCI (Cross-Channel Interference) according to equation (5) from the paper.
        
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
        running_services = topology[current_node][next_node]['running_services']
        d_l = len(running_services)
        
        # If no other services on this link, XCI = 0
        if d_l <= 1:  # Only current service or no services
            return 0.0
        
        # Main factor: Nl * (16/81) * (γ²P³) / (π²α²)
        main_factor = n_spans * (16/81) * (self.physical_params.gamma**2 * 
                                        self.physical_params.launch_power**3) / \
                    (pi**2 * self.physical_params.alpha**2)
        
        p_xci = 0.0
        
        # Sum over all interfering services (r' ≠ r)
        for other_service in running_services:
            if other_service.service_id == service.service_id:
                continue  # Skip self-interference
            
            # Calculate cross-dispersion parameter φ_r,r'
            phi_r_r_prime = self.calculate_dispersion_params(
                service.center_frequency, 
                other_service.center_frequency
            )
            
            # Check for zero division - skip if denominator too small
            denominator = phi_r_r_prime * other_service.bandwidth
            if abs(denominator) < 1e-12:
                continue  # Skip this interfering service
            
            # Calculate ISRS term for interfering service r'
            isrs_term_r_prime = (2*self.physical_params.alpha - 
                                d_l*self.physical_params.launch_power*
                                self.physical_params.cr*other_service.center_frequency)**2
            
            # Calculate terms for interfering service r'
            term1_r_prime = (isrs_term_r_prime - self.physical_params.alpha**2) / \
                            self.physical_params.alpha
            
            term2_r_prime = (4*self.physical_params.alpha**2 - isrs_term_r_prime) / \
                            (2*self.physical_params.alpha)
            
            # Calculate the inner sum factor: 1/(φ_r,r' * B_r')
            inner_sum_factor = 1 / (phi_r_r_prime * other_service.bandwidth)
            
            # Calculate atan terms with current service bandwidth B_r
            atan_term1 = atan((2*pi**2/self.physical_params.alpha) * 
                            phi_r_r_prime * service.bandwidth)
            
            atan_term2 = atan((pi**2/self.physical_params.alpha) * 
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
        """Calculate OSNR considering directed path"""
        # Calculate ASE noise
        #print("Service in OSNR method", service)
        p_ase = self.calculate_ase_noise(service, topology)
        #print("P_ase", p_ase)
        
        # Calculate total NLI noise
        p_nli = 0
        path = service.path.node_list
        
        # Iterate through directed path
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            
            # Calculate SCI and XCI for this directed edge
            p_nli += self.calculate_sci(service, current_node, next_node, topology)
            p_nli += self.calculate_xci(service, current_node, next_node, topology)
            
        # Calculate OSNR
        
        osnr = self.physical_params.launch_power / (p_ase + p_nli)
        #print("osnr", osnr)
        
        # Convert to dB
        osnr_db = 10 * log10(osnr)
        #print("osnr_db", osnr_db)
        
        return osnr_db