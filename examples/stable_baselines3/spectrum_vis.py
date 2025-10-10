"""
spectrum_visualizer.py

Visualize spectrum status after each allocation and deallocation.
Shows occupied/free slots across all links and bands.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os


class SpectrumVisualizer:
    """
    Real-time spectrum status visualizer for RMSA environment.
    """
    
    def __init__(self, env, save_dir='./spectrum_plots/', interactive=False):
        """
        Initialize visualizer.
        
        Args:
            env: RMSAEnv instance
            save_dir: Directory to save plots
            interactive: If True, show plots interactively (slower)
        """
        self.env = env
        self.save_dir = save_dir
        self.interactive = interactive
        self.step_count = 0
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get environment info
        self.num_edges = env.topology.number_of_edges()
        self.num_bands = env.num_bands
        self.num_slots = env.num_spectrum_resources
        
        # Band boundaries
        self.c_start, self.c_end = env.get_shift(0)
        if self.num_bands > 1:
            self.l_start, self.l_end = env.get_shift(1)
        
        # Create colormap
        # 0 = free (white), 1 = occupied (blue), 2 = guard (red)
        colors = ['white', 'steelblue', 'lightcoral']
        self.cmap = ListedColormap(colors)
        
        if interactive:
            plt.ion()
    
    def plot_spectrum(self, title_suffix="", event_type=None, service_id=None):
        """
        Plot current spectrum status.
        
        Args:
            title_suffix: Additional text for title
            event_type: 'allocation' or 'deallocation'
            service_id: Service ID that triggered the event
        """
        # Get spectrum status
        available_slots = self.env.topology.graph.get('available_slots', None)
        if available_slots is None:
            return
        
        # Create figure with subplots for each band
        if self.num_bands == 1:
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            axes = [ax]
        else:
            fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # Plot each band
        for band_idx in range(self.num_bands):
            ax = axes[band_idx] if self.num_bands > 1 else axes[0]
            
            # Get band range
            start, end = self.env.get_shift(band_idx)
            
            # Extract spectrum for this band
            band_spectrum = np.zeros((self.num_edges, end - start))
            
            for edge_idx in range(self.num_edges):
                offset = edge_idx + (self.num_edges * band_idx)
                # 0 = occupied, 1 = free -> invert for visualization
                band_spectrum[edge_idx, :] = 1 - available_slots[offset, start:end]
            
            # Plot heatmap
            im = ax.imshow(band_spectrum, cmap=self.cmap, aspect='auto', 
                          interpolation='nearest', vmin=0, vmax=1)
            
            # Formatting
            band_name = "C-band" if band_idx == 0 else "L-band"
            ax.set_title(f'{band_name} Spectrum Status', fontsize=14, fontweight='bold')
            ax.set_xlabel('Spectrum Slot Index', fontsize=12)
            ax.set_ylabel('Link Index', fontsize=12)
            
            # Grid
            ax.set_xticks(np.arange(0, end - start, 50))
            ax.set_yticks(np.arange(self.num_edges))
            ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
            cbar.ax.set_yticklabels(['Free', 'Occupied'])
            
            # Calculate utilization
            total_slots = self.num_edges * (end - start)
            occupied = np.sum(band_spectrum)
            util = (occupied / total_slots * 100) if total_slots > 0 else 0
            
            # Add utilization text
            ax.text(0.02, 0.98, f'Utilization: {util:.2f}%', 
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
        
        # Main title
        title = f"Step {self.step_count}"
        if event_type:
            title += f" - {event_type.capitalize()}"
        if service_id is not None:
            title += f" (Service {service_id})"
        if title_suffix:
            title += f" {title_suffix}"
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        filename = f'spectrum_step_{self.step_count:05d}.png'
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        
        if self.interactive:
            plt.pause(0.1)
        else:
            plt.close(fig)
        
        self.step_count += 1
    
    def plot_detailed_link(self, link_idx, band_idx):
        """
        Plot detailed view of a specific link and band.
        
        Args:
            link_idx: Link index
            band_idx: Band index
        """
        available_slots = self.env.topology.graph.get('available_slots', None)
        if available_slots is None:
            return
        
        start, end = self.env.get_shift(band_idx)
        offset = link_idx + (self.num_edges * band_idx)
        
        link_spectrum = available_slots[offset, start:end]
        
        fig, ax = plt.subplots(figsize=(16, 4))
        
        # Create visual representation
        spectrum_visual = 1 - link_spectrum  # Invert for visualization
        spectrum_2d = np.repeat(spectrum_visual[np.newaxis, :], 10, axis=0)
        
        im = ax.imshow(spectrum_2d, cmap=self.cmap, aspect='auto', 
                      interpolation='nearest')
        
        band_name = "C-band" if band_idx == 0 else "L-band"
        ax.set_title(f'Link {link_idx} - {band_name} Detailed View', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Spectrum Slot Index', fontsize=12)
        ax.set_yticks([])
        
        # Grid every 10 slots
        ax.set_xticks(np.arange(0, end - start, 10), minor=False)
        ax.set_xticks(np.arange(0, end - start, 1), minor=True)
        ax.grid(True, which='major', color='black', linewidth=0.8)
        ax.grid(True, which='minor', color='gray', linewidth=0.3, alpha=0.5)
        
        plt.colorbar(im, ax=ax, ticks=[0, 1]).ax.set_yticklabels(['Free', 'Occupied'])
        
        plt.tight_layout()
        filename = f'link_{link_idx}_band_{band_idx}_step_{self.step_count:05d}.png'
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def plot_fragmentation_analysis(self):
        """Plot fragmentation analysis showing block sizes."""
       
        from optical_rl_gym.envs import RMSAEnv
        rle = RMSAEnv.rle
        
        fig, axes = plt.subplots(self.num_bands, 1, figsize=(14, 6*self.num_bands))
        if self.num_bands == 1:
            axes = [axes]
        
        for band_idx in range(self.num_bands):
            ax = axes[band_idx]
            start, end = self.env.get_shift(band_idx)
            
            # Collect all available block sizes across all links
            all_block_sizes = []
            
            for edge_idx in range(self.num_edges):
                offset = edge_idx + (self.num_edges * band_idx)
                available_slots = self.env.topology.graph['available_slots']
                link_spectrum = available_slots[offset, start:end]
                
                # Get blocks using RLE
                positions, values, lengths = rle(link_spectrum)
                available_indices = np.where(values == 1)[0]
                
                if len(available_indices) > 0:
                    block_sizes = lengths[available_indices]
                    all_block_sizes.extend(block_sizes)
            
            if all_block_sizes:
                # Plot histogram
                ax.hist(all_block_sizes, bins=50, edgecolor='black', alpha=0.7)
                
                band_name = "C-band" if band_idx == 0 else "L-band"
                ax.set_title(f'{band_name} Available Block Size Distribution', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Block Size (slots)', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                stats_text = (f'Total Blocks: {len(all_block_sizes)}\n'
                            f'Mean Size: {np.mean(all_block_sizes):.1f}\n'
                            f'Max Size: {np.max(all_block_sizes)}\n'
                            f'Min Size: {np.min(all_block_sizes)}')
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        filename = f'fragmentation_analysis_step_{self.step_count:05d}.png'
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    def close(self):
        """Close visualizer."""
        if self.interactive:
            plt.ioff()
            plt.close('all')


# =============================================================================
# Integration Helper
# =============================================================================

def visualize_ksp_episode(env, agent, num_steps=50, save_dir='./spectrum_plots/'):
    """
    Run episode with spectrum visualization after each step.
    
    Args:
        env: RMSAEnv instance
        agent: Agent instance (e.g., KSP_FirstFit_Agent)
        num_steps: Number of steps to visualize
        save_dir: Directory to save plots
    """
    visualizer = SpectrumVisualizer(env, save_dir=save_dir, interactive=False)
    
    print(f"Visualizing {num_steps} steps...")
    print(f"Saving plots to: {save_dir}")
    
    obs = env.reset()
    
    # Plot initial state
    visualizer.plot_spectrum(title_suffix="Initial State")
    
    for step in range(num_steps):
        # Get action
        action = agent.select_action(obs)
        
        # Execute
        obs, reward, done, info = env.step(action)
        
        # Determine event type
        if env.current_service.accepted:
            event = "allocation"
            service_id = env.current_service.service_id
        else:
            event = "rejection"
            service_id = env.current_service.service_id
        
        # Plot after allocation/rejection
        visualizer.plot_spectrum(
            event_type=event,
            service_id=service_id
        )
        
        # Every 10 steps, plot fragmentation analysis
        if (step + 1) % 10 == 0:
            visualizer.plot_fragmentation_analysis()
        
        if done:
            break
        
        print(f"Step {step+1}/{num_steps}: {event} - Service {service_id} - Reward: {reward:.2f}")
    
    visualizer.close()
    print(f"\nVisualization complete! Saved {visualizer.step_count} plots to {save_dir}")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import networkx as nx
    from optical_rl_gym.envs import RMSAEnv
    from ksp_ff import KSP_FirstFit_Agent
    
    # Load topology
    topology = nx.read_graphml('./topologies/nsfnet.graphml', node_type=int)
    
    # Create environment
    env = RMSAEnv(
        topology=topology,
        seed=42,
        episode_length=100,
        num_bands=2,
        bit_rates=[50, 100, 200],
        k_paths=5,
        load=100.0
    )
    
    # Create agent
    agent = KSP_FirstFit_Agent(env)
    
    # Visualize 30 steps
    visualize_ksp_episode(
        env=env,
        agent=agent,
        num_steps=30,
        save_dir='./spectrum_plots/'
    )