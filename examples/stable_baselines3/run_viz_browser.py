"""
live_spectrum_visualizer.py

Enhanced live browser-based spectrum visualization for RMSA environments.
Works with any agent (KSP-FF, RL, etc.) and any RMSA environment.

Requirements:
    pip install flask flask-socketio simple-websocket
"""

import numpy as np
import json
from flask import Flask, render_template_string
from flask_socketio import SocketIO
import threading
import webbrowser
import time


class LiveSpectrumVisualizer:
    """Real-time browser-based spectrum visualizer with comprehensive metrics."""
    
    def __init__(self, env, port=5000, auto_open=True):
        """
        Initialize live visualizer.
        
        Args:
            env: RMSA environment instance
            port: Port for web server
            auto_open: Automatically open browser
        """
        self.env = env
        self.port = port
        self.auto_open = auto_open
        
        # Environment info
        self.num_edges = env.topology.number_of_edges()
        self.num_bands = env.num_bands
        self.num_slots = env.num_spectrum_resources
        
        # Band boundaries
        self.band_info = []
        for band_idx in range(self.num_bands):
            start, end = env.get_shift(band_idx)
            band_name = "C-band" if band_idx == 0 else "L-band"
            self.band_info.append({
                'name': band_name,
                'start': int(start),
                'end': int(end),
                'capacity': int(end - start)
            })
        
        # Statistics
        self.step_count = 0
        self.last_service = None
        
        # Reference to agent (will be set externally)
        self.agent = None
        
        # Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'spectrum-viz-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Setup routes
        self._setup_routes()
        
        # Server thread
        self.server_thread = None
        self.running = False
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"Client connected to visualization")
            self.socketio.emit('config', {
                'num_edges': self.num_edges,
                'num_bands': self.num_bands,
                'band_info': self.band_info
            })
            self._send_update()
        
        @self.socketio.on('request_update')
        def handle_request():
            self._send_update()
    
    def _calculate_network_capacity(self):
        """Calculate total and remaining network capacity in Gbps."""
        # Total capacity calculation
        total_capacity = 0
        remaining_capacity = 0
        
        for band_idx in range(self.num_bands):
            start, end = self.env.get_shift(band_idx)
            band_slots = end - start
            
            # Each slot = 12.5 GHz bandwidth
            # With efficient modulation (PM_16QAM), each slot can carry ~200 Gbps
            # Conservative estimate: average 100 Gbps per slot
            gbps_per_slot = 100
            
            total_band_capacity = self.num_edges * band_slots * gbps_per_slot
            total_capacity += total_band_capacity
            
            # Calculate free slots
            available_slots = self.env.topology.graph.get('available_slots')
            if available_slots is not None:
                free_slots = np.sum(available_slots[
                    band_idx * self.num_edges:(band_idx + 1) * self.num_edges,
                    start:end
                ])
                remaining_capacity += free_slots * gbps_per_slot
        
        return total_capacity, remaining_capacity
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        # Check for numpy integer types (NumPy 2.0 compatible)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        # Check for numpy floating types (NumPy 2.0 compatible)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        return obj
    
    def _send_update(self):
        """Send comprehensive spectrum update to browser."""
        try:
            available_slots = self.env.topology.graph.get('available_slots')
            if available_slots is None:
                return
            
            # Prepare data for each band
            bands_data = []
            
            for band_idx in range(self.num_bands):
                start, end = self.env.get_shift(band_idx)
                band_capacity = end - start
                
                # Extract spectrum
                band_spectrum = []
                for edge_idx in range(self.num_edges):
                    offset = edge_idx + (self.num_edges * band_idx)
                    link_slots = (1 - available_slots[offset, start:end]).tolist()
                    band_spectrum.append(link_slots)
                
                # Calculate utilization
                total_slots = self.num_edges * band_capacity
                occupied = np.sum(1 - available_slots[
                    band_idx * self.num_edges:(band_idx + 1) * self.num_edges,
                    start:end
                ])
                utilization = float(occupied / total_slots * 100) if total_slots > 0 else 0.0
                
                # Calculate fragmentation for this band
                band_frag = self.env.topology.graph.get("current_band_fragmentation", {}).get(band_idx, 0.0)
                
                bands_data.append({
                    'index': band_idx,
                    'name': self.band_info[band_idx]['name'],
                    'spectrum': band_spectrum,
                    'utilization': round(utilization, 2),
                    'fragmentation': round(band_frag, 2)
                })
            
            # Network-wide fragmentation
            network_frag = self.env.topology.graph.get("current_network_fragmentation", 0.0)
            
            # Active services metrics
            running_services = self.env.topology.graph.get("running_services", [])
            active_count = len(running_services)
            
            # Total active bit rate
            total_bit_rate = sum(s.bit_rate for s in running_services if hasattr(s, 'bit_rate'))
            
            # Average OSNR margin
            osnr_margins = [s.OSNR_margin for s in running_services if hasattr(s, 'OSNR_margin')]
            avg_osnr = round(np.mean(osnr_margins), 2) if osnr_margins else 0.0
            
            # Network capacity
            total_capacity, remaining_capacity = self._calculate_network_capacity()
            
            # Blocking reasons - USE AGENT'S ACCUMULATED COUNTERS
            # This captures ALL blocking reasons encountered during search,
            # not just the final reason per service
            blocking_reasons = {}
            
            # Use agent's accumulated counters if available
            if hasattr(self, 'agent') and self.agent is not None and hasattr(self.agent, 'blocking_reasons_encountered'):
                for reason in self.agent.blocking_reasons_encountered:
                    blocking_reasons[reason] = int(self.agent.blocking_reasons_encountered[reason])
            elif hasattr(self.env, 'blocking_reasons'):
                # Fallback to env's tracking if no agent
                for reason, count in self.env.blocking_reasons.items():
                    if reason not in ['total_blocked', 'total_accepted']:
                        blocking_reasons[reason] = int(count)
            
            # Calculate network load in Erlang
            # Load (Erlang) = arrival_rate × holding_time
            # Or approximately: number of active services
            if hasattr(self.env, 'mean_service_holding_time') and hasattr(self.env, 'mean_service_inter_arrival_time'):
                arrival_rate = 1.0 / self.env.mean_service_inter_arrival_time
                holding_time = self.env.mean_service_holding_time
                network_load_erlang = arrival_rate * holding_time
            else:
                # Fallback: use number of active services as approximation
                network_load_erlang = float(active_count)
            
            # Environment statistics
            stats = {
                'step': self.step_count,
                'services_processed': self.env.services_processed,
                'services_accepted': self.env.services_accepted,
                'blocking_rate': round(
                    (self.env.services_processed - self.env.services_accepted) / 
                    self.env.services_processed * 100, 2
                ) if self.env.services_processed > 0 else 0.0,
                'network_fragmentation': round(network_frag, 2),
                'active_services': active_count,
                'total_bit_rate': round(total_bit_rate, 2),
                'avg_osnr_margin': avg_osnr,
                'total_capacity_gbps': round(total_capacity, 2),
                'remaining_capacity_gbps': round(remaining_capacity, 2),
                'network_load_erlang': round(network_load_erlang, 2),
                'blocking_reasons': blocking_reasons
            }
            
            # Last service info (FIXED: Get from services list, not current_service)
            service_info = None
            if self.last_service:
                service_info = {
                    'id': int(self.last_service.service_id),
                    'source': int(self.last_service.source),
                    'destination': int(self.last_service.destination),
                    'bit_rate': float(self.last_service.bit_rate),
                    'accepted': bool(self.last_service.accepted)
                }
                
                if self.last_service.accepted and hasattr(self.last_service, 'path'):
                    service_info['band'] = int(self.last_service.band)
                    service_info['initial_slot'] = int(self.last_service.initial_slot)
                    service_info['num_slots'] = int(self.last_service.number_slots)
                    if hasattr(self.last_service, 'modulation_format'):
                        service_info['modulation'] = str(self.last_service.modulation_format)
                    if hasattr(self.last_service, 'OSNR_margin'):
                        service_info['osnr_margin'] = float(round(self.last_service.OSNR_margin, 2))
                elif hasattr(self.last_service, 'blocking_reason'):
                    service_info['blocking_reason'] = str(self.last_service.blocking_reason)
            
            # Convert all data to JSON-serializable format
            data_to_send = self._convert_to_json_serializable({
                'bands': bands_data,
                'stats': stats,
                'service': service_info
            })
            
            # Send update
            self.socketio.emit('spectrum_update', data_to_send)
            
        except Exception as e:
            print(f"Error sending update: {e}")
            import traceback
            traceback.print_exc()
    
    def update(self, service=None):
        """
        Send update to browser.
        Call this after each env.step() or env.reset().
        
        Args:
            service: Optional service object
        """
        self.step_count += 1
        
        # FIXED: Get the LAST PROCESSED service from services list
        if service is None:
            if hasattr(self.env.topology, 'graph') and 'services' in self.env.topology.graph:
                services = self.env.topology.graph['services']
                if len(services) > 0:
                    self.last_service = services[-1]
            else:
                self.last_service = None
        else:
            self.last_service = service
        
        if self.running:
            self._send_update()
    
    def set_agent(self, agent):
        """Set the agent reference for tracking blocking reasons."""
        self.agent = agent
    
    def start(self, blocking=False):
        """Start the visualization server."""
        if self.running:
            print("Visualizer already running")
            return
        
        self.running = True
        
        def run_server():
            print(f"\n{'='*60}")
            print(f"Live Spectrum Visualizer")
            print(f"{'='*60}")
            print(f"Server starting on http://localhost:{self.port}")
            print(f"Open this URL in your browser to view live spectrum")
            print(f"{'='*60}\n")
            
            if self.auto_open:
                time.sleep(1)
                webbrowser.open(f'http://localhost:{self.port}')
            
            self.socketio.run(self.app, host='0.0.0.0', port=self.port,
                            debug=False, use_reloader=False, log_output=False)
        
        if blocking:
            run_server()
        else:
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            time.sleep(2)
    
    def stop(self):
        """Stop the visualization server."""
        self.running = False
        if self.server_thread:
            print("\nStopping visualization server...")


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Spectrum Visualization</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            padding: 15px;
            overflow-x: hidden;
        }
        .container { max-width: 1900px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 15px;
            color: #4CAF50;
            font-size: 1.8em;
        }
        
        /* Grid Layout */
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 15px;
            margin-bottom: 15px;
        }
        
        /* Left Column - Main Content */
        .main-content { display: flex; flex-direction: column; gap: 15px; }
        
        /* Right Column - Fixed Metrics */
        .metrics-sidebar {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        /* Stats Cards */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
        }
        .stat-card {
            background: #1a1a1a;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #333;
        }
        .stat-label {
            color: #888;
            font-size: 0.85em;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 1.4em;
            font-weight: bold;
            color: #4CAF50;
        }
        
        /* Service Info - COMPACT FIXED SIZE */
        .service-info {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 10px;
            height: 185px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .service-info h3 {
            color: #4CAF50;
            margin-bottom: 8px;
            font-size: 1em;
        }
        .service-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 4px;
            font-size: 0.8em;
            flex: 1;
        }
        .service-item {
            background: #0a0a0a;
            padding: 4px 6px;
            border-radius: 4px;
            line-height: 1.3;
        }
        .service-item strong { color: #888; }
        .accepted { color: #4CAF50; font-weight: bold; }
        .rejected { color: #f44336; font-weight: bold; }
        
        /* Network Metrics Panel */
        .network-metrics {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 12px;
        }
        .network-metrics h3 {
            color: #4CAF50;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #222;
            font-size: 0.9em;
        }
        .metric-row:last-child { border-bottom: none; }
        .metric-label { color: #888; }
        .metric-value { color: #4CAF50; font-weight: bold; }
        
        /* Blocking Reasons */
        .blocking-reasons {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 12px;
            max-height: 250px;
            overflow-y: auto;
        }
        .blocking-reasons h3 {
            color: #4CAF50;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .reason-item {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            font-size: 0.85em;
            border-bottom: 1px solid #222;
        }
        .reason-item:last-child { border-bottom: none; }
        .reason-name { color: #ddd; }
        .reason-count {
            background: #f44336;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: bold;
            min-width: 35px;
            text-align: center;
            display: inline-block;
        }
        .reason-count-zero {
            background: #2a2a2a;
            color: #666;
            padding: 2px 8px;
            border-radius: 3px;
            min-width: 35px;
            text-align: center;
            display: inline-block;
        }
        
        /* Band Container */
        .band-container {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 15px;
            border-radius: 8px;
        }
        .band-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .band-title { font-size: 1.2em; color: #4CAF50; }
        .band-stats {
            display: flex;
            gap: 15px;
            font-size: 0.9em;
        }
        .band-stat {
            background: #0a0a0a;
            padding: 5px 12px;
            border-radius: 5px;
        }
        .band-stat-label { color: #888; }
        .band-stat-value { color: #4CAF50; font-weight: bold; }
        
        canvas {
            width: 100%;
            height: 300px;
            border: 2px solid #333;
            border-radius: 5px;
            background: #000;
        }
        
        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 10px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
        }
        .legend-color {
            width: 25px;
            height: 18px;
            border: 1px solid #666;
            border-radius: 3px;
        }
        .free { background: #000; }
        .occupied { background: #2196F3; }
        
        .status {
            text-align: center;
            padding: 8px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 5px;
            margin-top: 15px;
            color: #888;
            font-size: 0.9em;
        }
        .connected { color: #4CAF50; }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0a0a0a; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #444; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Live Spectrum Visualization</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Step</div>
                <div class="stat-value" id="step">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Processed</div>
                <div class="stat-value" id="processed">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Accepted</div>
                <div class="stat-value" id="accepted">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Blocking Rate</div>
                <div class="stat-value" id="blocking">0%</div>
            </div>
        </div>
        
        <div class="dashboard">
            <div class="main-content">
                <div id="bands"></div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color free"></div>
                        <span>Free Slots</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color occupied"></div>
                        <span>Occupied Slots</span>
                    </div>
                </div>
            </div>
            
            <div class="metrics-sidebar">
                <div class="service-info">
                    <h3>Last Service</h3>
                    <div class="service-grid" id="serviceDetails">
                        <div class="service-item" style="grid-column: 1/-1; text-align: center; color: #666;">
                            Waiting for first service...
                        </div>
                    </div>
                </div>
                
                <div class="network-metrics">
                    <h3>Network Metrics</h3>
                    <div class="metric-row">
                        <span class="metric-label">Active Services</span>
                        <span class="metric-value" id="activeServices">0</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Network Load</span>
                        <span class="metric-value" id="networkLoad">0.0%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Total Bit Rate</span>
                        <span class="metric-value" id="totalBitRate">0 Gbps</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Avg OSNR Margin</span>
                        <span class="metric-value" id="avgOSNR">0.0 dB</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Network Fragmentation</span>
                        <span class="metric-value" id="netFrag">0.0</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Remaining Capacity</span>
                        <span class="metric-value" id="remainingCap">0 Gbps</span>
                    </div>
                </div>
                
                <div class="blocking-reasons">
                    <h3>Blocking Reasons</h3>
                    <div id="blockingReasons">
                        <div style="text-align: center; color: #666; padding: 20px;">
                            No blocking events yet
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="status">
            <span id="status">Connecting to server...</span>
        </div>
    </div>

    <script>
        const socket = io();
        let config = null;
        let canvases = {};
        
        socket.on('connect', () => {
            document.getElementById('status').innerHTML = 
                '<span class="connected">● Connected</span>';
        });
        
        socket.on('disconnect', () => {
            document.getElementById('status').innerHTML = 
                '● Disconnected';
        });
        
        socket.on('config', (data) => {
            config = data;
            setupBands(data);
        });
        
        socket.on('spectrum_update', (data) => {
            updateStats(data.stats);
            updateService(data.service);
            updateSpectrum(data.bands);
        });
        
        function setupBands(cfg) {
            const bandsDiv = document.getElementById('bands');
            bandsDiv.innerHTML = '';
            
            cfg.band_info.forEach((band, idx) => {
                const bandDiv = document.createElement('div');
                bandDiv.className = 'band-container';
                bandDiv.innerHTML = `
                    <div class="band-header">
                        <div class="band-title">${band.name}</div>
                        <div class="band-stats">
                            <div class="band-stat">
                                <span class="band-stat-label">Util: </span>
                                <span class="band-stat-value" id="util-${idx}">0.00%</span>
                            </div>
                            <div class="band-stat">
                                <span class="band-stat-label">Frag: </span>
                                <span class="band-stat-value" id="frag-${idx}">0.0</span>
                            </div>
                        </div>
                    </div>
                    <canvas id="canvas-${idx}"></canvas>
                `;
                bandsDiv.appendChild(bandDiv);
                canvases[idx] = document.getElementById(`canvas-${idx}`);
            });
        }
        
        function updateStats(stats) {
            document.getElementById('step').textContent = stats.step;
            document.getElementById('processed').textContent = stats.services_processed;
            document.getElementById('accepted').textContent = stats.services_accepted;
            document.getElementById('blocking').textContent = stats.blocking_rate + '%';
            
            document.getElementById('activeServices').textContent = stats.active_services;
            document.getElementById('networkLoad').textContent = stats.network_load_erlang + ' E';
            document.getElementById('totalBitRate').textContent = stats.total_bit_rate + ' Gbps';
            document.getElementById('avgOSNR').textContent = stats.avg_osnr_margin + ' dB';
            document.getElementById('netFrag').textContent = stats.network_fragmentation;
            document.getElementById('remainingCap').textContent = stats.remaining_capacity_gbps + ' Gbps';
            
            // Update blocking reasons - show ALL reasons
            const reasonsDiv = document.getElementById('blockingReasons');
            const blockingReasons = stats.blocking_reasons || {};
            
            // Define all possible blocking reasons with better labels
            const reasonConfig = [
                { key: 'osnr_interference_violation', label: 'OSNR Interference' },
                { key: 'osnr_threshold_violation', label: 'OSNR Threshold' },
                { key: 'spectrum_unavailable', label: 'Spectrum Unavailable' },
                { key: 'no_modulation_available', label: 'No Modulation' },
                { key: 'path_length_exceeded', label: 'Path Too Long' },
                { key: 'invalid_action', label: 'No Valid Allocation' }
            ];
            
            let html = '';
            let hasAnyBlocks = false;
            
            // Show all reasons
            for (const config of reasonConfig) {
                const count = blockingReasons[config.key] || 0;
                if (count > 0) hasAnyBlocks = true;
                
                const countClass = count > 0 ? 'reason-count' : 'reason-count-zero';
                html += `
                    <div class="reason-item">
                        <span class="reason-name">${config.label}</span>
                        <span class="${countClass}">${count}</span>
                    </div>
                `;
            }
            
            if (!hasAnyBlocks) {
                reasonsDiv.innerHTML = '<div style="text-align: center; color: #4CAF50; padding: 20px; font-size: 0.9em;">✓ No blocking events</div>';
            } else {
                reasonsDiv.innerHTML = html;
            }
        }
        
        function updateService(service) {
            const serviceDetails = document.getElementById('serviceDetails');
            
            if (!service) {
                serviceDetails.innerHTML = '<div class="service-item" style="grid-column: 1/-1; text-align: center; color: #666;">Waiting for service...</div>';
                return;
            }
            
            let html = `
                <div class="service-item"><strong>ID:</strong> ${service.id}</div>
                <div class="service-item"><strong>Route:</strong> ${service.source} → ${service.destination}</div>
                <div class="service-item"><strong>Bit Rate:</strong> ${service.bit_rate} Gbps</div>
                <div class="service-item">
                    <strong>Status:</strong> 
                    <span class="${service.accepted ? 'accepted' : 'rejected'}">
                        ${service.accepted ? '✓ Accepted' : '✗ Rejected'}
                    </span>
                </div>
            `;
            
            if (service.accepted) {
                html += `
                    <div class="service-item"><strong>Band:</strong> ${service.band === 0 ? 'C-band' : 'L-band'}</div>
                    <div class="service-item"><strong>Slot:</strong> ${service.initial_slot}</div>
                    <div class="service-item"><strong>Slots:</strong> ${service.num_slots}</div>
                `;
                if (service.modulation) {
                    html += `<div class="service-item"><strong>Mod:</strong> ${service.modulation}</div>`;
                }
                if (service.osnr_margin !== undefined) {
                    html += `<div class="service-item"><strong>OSNR:</strong> ${service.osnr_margin} dB</div>`;
                }
            } else if (service.blocking_reason) {
                const displayReason = service.blocking_reason.replace(/_/g, ' ');
                html += `<div class="service-item" style="grid-column: 1/-1;"><strong>Reason:</strong> ${displayReason}</div>`;
            }
            
            serviceDetails.innerHTML = html;
        }
        
        function updateSpectrum(bands) {
            bands.forEach(band => {
                const canvas = canvases[band.index];
                if (!canvas) return;
                
                const ctx = canvas.getContext('2d');
                const spectrum = band.spectrum;
                
                document.getElementById(`util-${band.index}`).textContent = band.utilization + '%';
                document.getElementById(`frag-${band.index}`).textContent = band.fragmentation;
                
                const numLinks = spectrum.length;
                const numSlots = spectrum[0].length;
                
                canvas.width = numSlots * 3;
                canvas.height = numLinks * 10;
                
                for (let link = 0; link < numLinks; link++) {
                    for (let slot = 0; slot < numSlots; slot++) {
                        ctx.fillStyle = spectrum[link][slot] ? '#2196F3' : '#000000';
                        ctx.fillRect(slot * 3, link * 10, 3, 10);
                    }
                }
            });
        }
    </script>
</body>
</html>
"""


def run_with_visualization(env, agent, num_episodes=10, update_delay=0.1):
    """Run agent with live visualization."""
    visualizer = LiveSpectrumVisualizer(env, port=5001)
    visualizer.set_agent(agent)  # Set agent reference for blocking reason tracking
    visualizer.start(blocking=False)
    
    print("\nStarting simulation with live visualization...")
    print("Open http://localhost:5000 in your browser\n")
    
    time.sleep(2)
    
    for episode in range(num_episodes):
        obs = env.reset()
        visualizer.update()
        
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            visualizer.update()
            
            episode_reward += reward
            time.sleep(update_delay)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Blocking={info.get('episode_service_blocking_rate', 0):.4f}")
    
    print("\nSimulation complete! Server still running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        visualizer.stop()


if __name__ == "__main__":
    import os
    import pickle
    from optical_rl_gym.envs import RMSAEnv
    from ksp_ff_osnr_check import KSP_FF_Agent
    
    topology_path = '../topologies/indian_net_5-paths_new.h5'
    
    if os.path.exists(topology_path):
        with open(topology_path, 'rb') as f:
            topology = pickle.load(f)
    else:
        print(f"Topology not found: {topology_path}")
        exit(1)
    
    env = RMSAEnv(
        topology=topology,
        seed=42,
        episode_length=100,
        num_bands=2,
        bit_rates=[50, 100, 200],
        k_paths=5,
        mean_service_holding_time=50.0,
        mean_service_inter_arrival_time=0.1
    )
    
    agent = KSP_FF_Agent(env, debug=True)
    
    # Option 1: Use helper function (automatically sets agent)
    run_with_visualization(env, agent, num_episodes=5, update_delay=0.1)
