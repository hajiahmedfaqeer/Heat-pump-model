
"""
Heat Pump system modeling and analysis with TESPy.

This script:
  1) Builds a TESPy network of a simple heat pump (compressor–condenser–valve–evaporator)
  2) Solves a design point
  3) Runs off-design scenarios and a time-series simulation (with analytical fallback)
  4) Plots and saves results

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tespy.components import (Source, Sink, Compressor, Condenser, Pump, HeatExchanger, Valve)
from tespy.connections import Connection, Bus
from tespy.networks import Network
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HeatPumpModel:
    """Wraps the TESPy heat pump network and utility methods."""

    def __init__(self):
        # Holders for TESPy objects and flags
        self.network = None
        self.components = {}
        self.connections = {}
        self.buses = {}
        self.design_solved = False

        # Build the base network
        self.setup_network()
        
    def setup_network(self):
        """Create network, components, and wire them up."""
        # Network uses water (secondary) and R134a (refrigerant)
        self.network = Network(
            fluids=['water', 'R134a'],
            T_unit='C', p_unit='bar', h_unit='kJ/kg', m_unit='kg/s')
        
        # --- Components in the refrigeration cycle ---
        self.components['compressor'] = Compressor('Compressor')
        self.components['condenser'] = Condenser('Condenser')
        self.components['valve'] = Valve('Expansion Valve')
        self.components['evaporator'] = HeatExchanger('Evaporator')

        # --- Pumps for source/sink side water loops ---
        self.components['source_pump'] = Pump('Source Pump')
        self.components['sink_pump'] = Pump('Sink Pump')
        
        # --- Water side boundaries (heat source loop) ---
        self.components['source_in'] = Source('Source In')
        self.components['source_out'] = Sink('Source Out')
        
        # --- Water side boundaries (heat sink loop) ---
        self.components['sink_in'] = Source('Sink In')
        self.components['sink_out'] = Sink('Sink Out')
        
        # --- Refrigerant boundaries (open cycle for setup convenience) ---
        self.components['refrigerant_source'] = Source('Refrigerant Source')
        self.components['refrigerant_sink'] = Sink('Refrigerant Sink')
        
        # Create all pipes between components and add to the network
        self.setup_connections()

        # Define energy buses (power and heat flows)
        self.setup_buses()
        
    def setup_connections(self):
        """Connect all components with TESPy Connections."""

        # --- Refrigerant loop: source -> compressor -> condenser -> valve -> evaporator -> sink ---
        self.connections['c1'] = Connection(
            self.components['refrigerant_source'], 'out1',
            self.components['compressor'], 'in1', label='c1')
        
        self.connections['c2'] = Connection(
            self.components['compressor'], 'out1',
            self.components['condenser'], 'in1', label='c2')
        
        self.connections['c3'] = Connection(
            self.components['condenser'], 'out1',
            self.components['valve'], 'in1', label='c3')
        
        self.connections['c4'] = Connection(
            self.components['valve'], 'out1',
            self.components['evaporator'], 'in1', label='c4')
        
        self.connections['c5'] = Connection(
            self.components['evaporator'], 'out1',
            self.components['refrigerant_sink'], 'in1', label='c5')
        
        # --- Heat source water loop across evaporator secondary side ---
        self.connections['s1'] = Connection(
            self.components['source_in'], 'out1',
            self.components['source_pump'], 'in1', label='s1')
        
        self.connections['s2'] = Connection(
            self.components['source_pump'], 'out1',
            self.components['evaporator'], 'in2', label='s2')
        
        self.connections['s3'] = Connection(
            self.components['evaporator'], 'out2',
            self.components['source_out'], 'in1', label='s3')
        
        # --- Heat sink water loop across condenser secondary side ---
        self.connections['h1'] = Connection(
            self.components['sink_in'], 'out1',
            self.components['sink_pump'], 'in1', label='h1')
        
        self.connections['h2'] = Connection(
            self.components['sink_pump'], 'out1',
            self.components['condenser'], 'in2', label='h2')
        
        self.connections['h3'] = Connection(
            self.components['condenser'], 'out2',
            self.components['sink_out'], 'in1', label='h3')
        
        # Register all connections with the network
        for conn in self.connections.values():
            self.network.add_conns(conn)

    def setup_buses(self):
        """Set up energy buses to measure power and heat rates."""
        # Electrical/mechanical power consumers (positive on the bus)
        self.buses['power'] = Bus('Power Consumption')
        self.buses['power'].add_comps(
            {'comp': self.components['compressor'], 'base': 'bus'},
            {'comp': self.components['source_pump'], 'base': 'bus'},
            {'comp': self.components['sink_pump'], 'base': 'bus'})
        
        # Useful heat delivered at the condenser
        self.buses['heat_output'] = Bus('Heat Output')
        self.buses['heat_output'].add_comps(
            {'comp': self.components['condenser'], 'base': 'component'})
        
        # Heat extracted in the evaporator (for bookkeeping)
        self.buses['heat_input'] = Bus('Heat Input')
        self.buses['heat_input'].add_comps(
            {'comp': self.components['evaporator'], 'base': 'component'})
        
        # Add buses to network
        self.network.add_busses(self.buses['power'], self.buses['heat_output'], self.buses['heat_input'])
    
    def set_design_conditions(self):
        """
        Specify a design point (temperatures, pressures, flows, efficiencies)
        and solve the network. Falls back to a simplified setup on failure.
        """
        # Reset all connection attributes first
        for conn in self.connections.values():
            conn.set_attr(T=None, p=None, h=None, m=None, x=None)
        
        # Fix working fluids
        self.connections['s1'].set_attr(fluid={'water': 1})
        self.connections['h1'].set_attr(fluid={'water': 1})
        self.connections['c1'].set_attr(fluid={'R134a': 1})
        
        # Target specs (from problem statement)
        # Heat source: 40°C → 10°C, ~1000 kW
        # Heat sink:  40°C → 90°C, ~1012 kW
        
        # --- Source side (evaporator secondary) boundary conditions ---
        self.connections['s1'].set_attr(T=40, p=2)  # inlet to source loop
        self.connections['s3'].set_attr(T=10)       # outlet from source loop
        
        # --- Sink side (condenser secondary) boundary conditions ---
        self.connections['h1'].set_attr(T=40, p=2)  # inlet to sink loop
        self.connections['h3'].set_attr(T=90)       # outlet from sink loop
        
        # Calculate water mass flows from Q = m * cp * dT (cp ≈ 4.18 kJ/kgK)
        cp_water = 4.18  # kJ/kg·K
        m_source = 1000 / (cp_water * (40 - 10))  # kg/s
        m_sink = 1012 / (cp_water * (90 - 40))    # kg/s
        self.connections['s1'].set_attr(m=m_source)
        self.connections['h1'].set_attr(m=m_sink)
        
        # --- Refrigerant design guesses ---
        self.connections['c1'].set_attr(T=5, p=3)   # evap outlet (superheated vapor)
        self.connections['c3'].set_attr(p=20)       # condenser pressure (guess)
        self.connections['c4'].set_attr(x=0)        # valve outlet quality (2-phase)
        
        # --- Component efficiencies and pressure ratios ---
        self.components['compressor'].set_attr(eta_s=0.85)
        self.components['condenser'].set_attr(pr1=0.98, pr2=0.98)
        self.components['evaporator'].set_attr(pr1=0.98, pr2=0.98)
        self.components['source_pump'].set_attr(eta_s=0.75)
        self.components['sink_pump'].set_attr(eta_s=0.75)
        
        try:
            print("Solving design conditions...")
            self.network.solve('design')
            print("✓ Design solution successful!")
            self.design_solved = True
            self.print_design_results()
            return True
        except Exception as e:
            # If the detailed spec is too tight, try a simpler setup
            print(f"✗ Design solution failed: {e}")
            return self.set_simplified_design_conditions()
    
    def set_simplified_design_conditions(self):
        """Relaxed design spec to improve convergence if the main design fails."""
        try:
            # Rebuild a fresh network (cleans previous specs)
            self.network = Network(
                fluids=['water', 'R134a'],
                T_unit='C', p_unit='bar', h_unit='kJ/kg', m_unit='kg/s')
            
            self.setup_network()
            
            # Simpler fixed values (reasonable guesses)
            self.connections['s1'].set_attr(fluid={'water': 1}, T=40, p=2, m=8)
            self.connections['s3'].set_attr(T=10)
            self.connections['h1'].set_attr(fluid={'water': 1}, T=40, p=2, m=5)
            self.connections['h3'].set_attr(T=90)
            
            # Refrigerant side relaxed pressures and temperatures
            self.connections['c1'].set_attr(fluid={'R134a': 1}, T=8, p=4)
            self.connections['c3'].set_attr(p=18)
            
            # Slightly larger pressure losses
            self.components['compressor'].set_attr(eta_s=0.85)
            self.components['condenser'].set_attr(pr1=0.95, pr2=0.95)
            self.components['evaporator'].set_attr(pr1=0.95, pr2=0.95)
            
            self.network.solve('design')
            print("✓ Simplified design solution successful!")
            self.design_solved = True
            self.print_design_results()
            return True
        except Exception as e:
            print(f"✗ Simplified design also failed: {e}")
            return False
    
    def set_off_design_conditions(self, T_source_in, T_sink_in=40, load_factor=1.0):
        """
        Solve an off-design point.
        - T_source_in: source loop inlet temperature [°C]
        - T_sink_in: sink loop inlet temperature [°C]
        - load_factor: scales the water mass flows (part-load)
        """
        if not self.design_solved:
            print("Design must be solved first!")
            return None
            
        try:
            # Save design point values to restore later
            original_T_source = self.connections['s1'].T.val
            original_T_sink = self.connections['h1'].T.val
            original_m_source = self.connections['s1'].m.val
            original_m_sink = self.connections['h1'].m.val
            
            # Apply new boundary temps
            self.connections['s1'].set_attr(T=T_source_in)
            self.connections['h1'].set_attr(T=T_sink_in)
            
            # Scale mass flows to mimic part-load behavior
            if load_factor != 1.0:
                self.connections['s1'].set_attr(m=original_m_source * load_factor)
                self.connections['h1'].set_attr(m=original_m_sink * load_factor)
            
            # Solve at off-design with design as reference
            self.network.solve('offdesign', design_path='design')
            
            # Read results
            results = self.get_performance_results()
            
            # Restore design specs
            self.connections['s1'].set_attr(T=original_T_source)
            self.connections['h1'].set_attr(T=original_T_sink)
            self.connections['s1'].set_attr(m=original_m_source)
            self.connections['h1'].set_attr(m=original_m_sink)
            
            return results
            
        except Exception as e:
            print(f"Off-design simulation failed: {e}")
            return None
    
    def get_performance_results(self):
        """Collect main KPIs from the current solved state."""
        try:
            compressor_power = abs(self.buses['power'].P.val)       # kW
            heat_output = abs(self.buses['heat_output'].P.val)      # kW
            heat_input = abs(self.buses['heat_input'].P.val)        # kW
            cop = heat_output / compressor_power if compressor_power > 0 else 0
            
            return {
                'COP': cop,
                'compressor_power': compressor_power,
                'heat_output': heat_output,
                'heat_input': heat_input,
                'mass_flow_refrigerant': self.connections['c1'].m.val,
                'T_evap_out': self.connections['c1'].T.val,
                'T_cond_out': self.connections['c2'].T.val
            }
        except:
            # Return None if network is not solved or values are missing
            return None
    
    def print_design_results(self):
        """Pretty-print design results to the console."""
        print("\n" + "="*50)
        print("DESIGN CONDITION RESULTS")
        print("="*50)
        
        results = self.get_performance_results()
        if results:
            print(f"Compressor Power: {results['compressor_power']:.2f} kW")
            print(f"Heat Output (Condenser): {results['heat_output']:.2f} kW")
            print(f"Heat Input (Evaporator): {results['heat_input']:.2f} kW")
            print(f"COP: {results['COP']:.3f}")
            print(f"Compressor Efficiency: {self.components['compressor'].eta_s.val:.2f}")
            
            print(f"\nMass Flow Rates:")
            print(f"Source side: {self.connections['s1'].m.val:.3f} kg/s")
            print(f"Sink side: {self.connections['h1'].m.val:.3f} kg/s")
            print(f"Refrigerant: {self.connections['c1'].m.val:.3f} kg/s")
            
            print(f"\nKey Temperatures:")
            print(f"Source inlet: {self.connections['s1'].T.val:.1f} °C")
            print(f"Source outlet: {self.connections['s3'].T.val:.1f} °C")
            print(f"Sink inlet: {self.connections['h1'].T.val:.1f} °C")
            print(f"Sink outlet: {self.connections['h3'].T.val:.1f} °C")
            print(f"Compressor inlet: {self.connections['c1'].T.val:.1f} °C")
            print(f"Compressor outlet: {self.connections['c2'].T.val:.1f} °C")
        else:
            print("No results available")

import re

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names (lowercase, replace spaces/brackets/degree symbol)."""
    def fix(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"\s+", "_", s)                # spaces -> underscores
        s = s.replace("°", "deg")                 # degree symbol -> 'deg'
        s = s.replace("(", "").replace(")", "")   # remove parentheses
        s = s.replace("[", "_").replace("]", "")  # brackets -> underscores
        s = s.replace("__", "_").strip("_")       # collapse double underscores
        return s
    out = df.copy()
    out.columns = [fix(c) for c in df.columns]
    return out

def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Pick the first column containing any candidate substring (robust mapping)."""
    for cand in candidates:
        match = [c for c in df.columns if cand in c]
        if match:
            return match[0]
    return None

def load_real_data(path: str = "HP_case_data (2).xlsx") -> pd.DataFrame:
    """
    Load 'source' and 'sink' sheets from Excel, align by time, and return a tidy table with:
      timestamp, T_source_in, T_source_out, Flow_source, P_source,
                 T_sink_in,   T_sink_out,  Flow_sink,   P_sink,   Energy_sink,
                 T_source (alias of T_source_in)
    """
    print("Loading real dataset from Excel...")

    # Read both sheets at once
    xls = pd.read_excel(path, sheet_name=["source", "sink"])
    source = _normalize_columns(xls["source"])
    sink   = _normalize_columns(xls["sink"])

    # Flexible, name-agnostic column selection
    s_start = _pick(source, ["start_measurement", "start", "time, start", "time", "timestamp"])
    s_end   = _pick(source, ["end_measurement", "end", "time, end"])
    k_start = _pick(sink,   ["start_measurement", "start", "time, start", "time", "timestamp"])
    k_end   = _pick(sink,   ["end_measurement", "end", "time, end"])

    s_tin   = _pick(source, ["t_in_deg", "t_in", "tin_deg", "t_in[degc", "t_in_deg_c"])
    s_tout  = _pick(source, ["t_out_deg", "t_out", "tout_deg", "t_out[degc", "t_out_deg_c"])
    k_tin   = _pick(sink,   ["t_in_deg", "t_in", "tin_deg", "t_in[degc", "t_in_deg_c"])
    k_tout  = _pick(sink,   ["t_out_deg", "t_out", "tout_deg", "t_out[degc", "t_out_deg_c"])

    s_flow  = _pick(source, ["flow_kg/s", "mass_flow", "m_dot", "flow", "flow[kg/s]"])
    k_flow  = _pick(sink,   ["flow_kg/s", "mass_flow", "m_dot", "flow", "flow[kg/s]"])

    s_p     = _pick(source, ["p_bar", "pressure", "p", "p[bar]"])
    k_p     = _pick(sink,   ["p_bar", "pressure", "p", "p[bar]"])

    k_e     = _pick(sink,   ["energy_kwh", "energy", "q_kwh", "energy [kwh]"])

    def midpoint_ts(df, c_start, c_end):
        """Return start time or mid-time between start/end if both exist."""
        ts = pd.to_datetime(df[c_start], errors="coerce") if c_start else pd.NaT
        if c_end and c_end in df:
            te = pd.to_datetime(df[c_end], errors="coerce")
            ts = ts.where(te.isna(), ts + (te - ts) / 2)
        return ts

    # Build aligned source table
    src = pd.DataFrame({
        "timestamp":      midpoint_ts(source, s_start, s_end),
        "T_source_in":    source[s_tin]  if s_tin  else np.nan,
        "T_source_out":   source[s_tout] if s_tout else np.nan,
        "Flow_source":    source[s_flow] if s_flow else np.nan,
        "P_source":       source[s_p]    if s_p    else np.nan,
    }).dropna(subset=["timestamp"]).sort_values("timestamp")

    # Build aligned sink table
    snk = pd.DataFrame({
        "timestamp":      midpoint_ts(sink, k_start, k_end),
        "T_sink_in":      sink[k_tin]   if k_tin   else np.nan,
        "T_sink_out":     sink[k_tout]  if k_tout  else np.nan,
        "Flow_sink":      sink[k_flow]  if k_flow  else np.nan,
        "P_sink":         sink[k_p]     if k_p     else np.nan,
        "Energy_sink":    sink[k_e]     if k_e     else np.nan,
    }).dropna(subset=["timestamp"]).sort_values("timestamp")

    # Align rows by nearest timestamp (≤10 min)
    merged = pd.merge_asof(
        src, snk, on="timestamp", direction="nearest",
        tolerance=pd.Timedelta("10min")
    ).dropna(subset=["T_sink_in", "T_sink_out"], how="all")

    # Optional: hourly resample for smoother plots
    merged = (merged
              .set_index("timestamp").sort_index()
              .resample("1H").mean().interpolate()
              .reset_index())

    # Simulator uses this column name
    merged["T_source"] = merged["T_source_in"]

    # --- Unit conversions (uncomment if needed) ---
    # merged["Flow_source"] /= 3600.0  # kg/h -> kg/s
    # merged["Flow_sink"]   /= 3600.0
    # merged["P_source"]    /= 100.0   # kPa -> bar
    # merged["P_sink"]      /= 100.0

    print(f"Loaded {len(merged)} rows: {merged['timestamp'].min()} → {merged['timestamp'].max()}")
    return merged

def given_data(path: str = r"C:\Users\Saeed\OneDrive - Aalto University\Desktop\ai training\HP_case_data (2).xlsx") -> pd.DataFrame:
    """Convenience wrapper to load the workbook from a fixed path."""
    return load_real_data(path)

def simulate_time_series(heat_pump, df):
    """
    Run off-design simulations over a subset of timestamps (≤50 points).
    Returns a DataFrame with COP, power, and heat rates at full and 80% load.
    """
    print("\nSimulating time series performance...")
    
    results = []
    # Use evenly spaced indices to reduce runtime
    sample_indices = np.linspace(0, len(df)-1, min(50, len(df)), dtype=int)
    
    successful_simulations = 0
    
    for idx in sample_indices:
        row = df.iloc[idx]
        T_source = row['T_source']
        
        # Case 1: Normal operation (full load)
        result_normal = heat_pump.set_off_design_conditions(
            T_source_in=T_source, T_sink_in=40, load_factor=1.0
        )
        # Case 2: Part load (80%)
        result_part_load = heat_pump.set_off_design_conditions(
            T_source_in=T_source, T_sink_in=40, load_factor=0.8
        )
        
        # Keep only if both runs succeeded
        if result_normal and result_part_load:
            results.append({
                'timestamp': row['timestamp'],
                'T_source': T_source,
                'COP_normal': result_normal['COP'],
                'COP_part_load': result_part_load['COP'],
                'power_normal': result_normal['compressor_power'],
                'power_part_load': result_part_load['compressor_power'],
                'heat_output_normal': result_normal['heat_output'],
                'heat_output_part_load': result_part_load['heat_output'],
                'heat_input_normal': result_normal['heat_input'],
                'heat_input_part_load': result_part_load['heat_input']
            })
            successful_simulations += 1
    
    print(f"✓ Successfully simulated {successful_simulations} out of {len(sample_indices)} time points")
    
    if results:
        return pd.DataFrame(results)
    else:
        # If TESPy off-design fails, fall back to simple analytical estimates
        print("NOT successful simulations. Creating analytical results...")
        return create_analytical_results(df)

def create_analytical_results(df):
    """
    Analytical fallback using a fraction of Carnot COP and simple scaling
    to generate plausible performance trends.
    """
    print("Creating analytical performance estimates...")
    
    results = []
    sample_indices = np.linspace(0, len(df)-1, min(50, len(df)), dtype=int)
    
    for idx in sample_indices:
        row = df.iloc[idx]
        T_source = row['T_source']
        
        # Carnot COP for heating (absolute temperatures)
        T_source_k = T_source + 273.15
        T_sink_k = 40 + 273.15  # fixed sink inlet temp
        cop_carnot = T_sink_k / (T_sink_k - T_source_k) if T_sink_k > T_source_k else 0
        
        # Realistic COPs: 50–60% of Carnot and lower at part load
        cop_normal = cop_carnot * 0.55
        cop_part_load = cop_carnot * 0.50
        
        # Clamp to realistic ranges
        cop_normal = max(2.0, min(5.0, cop_normal))
        cop_part_load = max(1.8, min(4.5, cop_part_load))
        
        # Heat output scales with source temperature (very simplified)
        base_heat_output = 1000  # kW
        heat_output_normal = base_heat_output * (T_source / 40)
        heat_output_part_load = heat_output_normal * 0.8
        
        # Electrical power = Q / COP
        power_normal = heat_output_normal / cop_normal if cop_normal > 0 else 0
        power_part_load = heat_output_part_load / cop_part_load if cop_part_load > 0 else 0
        
        results.append({
            'timestamp': row['timestamp'],
            'T_source': T_source,
            'COP_normal': cop_normal,
            'COP_part_load': cop_part_load,
            'power_normal': power_normal,
            'power_part_load': power_part_load,
            'heat_output_normal': heat_output_normal,
            'heat_output_part_load': heat_output_part_load,
            'heat_input_normal': heat_output_normal - power_normal,  # Q_in = Q_out - W
            'heat_input_part_load': heat_output_part_load - power_part_load
        })
    
    return pd.DataFrame(results)

def create_performance_plots(df, simulation_results):
    """Create time-series and correlation plots for performance."""
    
    # 2x2 dashboard of core trends
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Heat Pump Performance Analysis', fontsize=16, fontweight='bold')
    
    # (1) Source temperature over time
    ax1 = axes[0, 0]
    ax1.plot(df['timestamp'], df['T_source'], 'b-', alpha=0.7, linewidth=1)
    ax1.set_ylabel('Source Temperature (°C)')
    ax1.set_title('Heat Source Temperature Profile')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # (2) COP comparison over time
    ax2 = axes[0, 1]
    if len(simulation_results) > 0:
        ax2.plot(simulation_results['timestamp'], simulation_results['COP_normal'], 
                 'g-', label='Full Load', linewidth=2)
        ax2.plot(simulation_results['timestamp'], simulation_results['COP_part_load'], 
                 'r-', label='80% Load', linewidth=2)
        ax2.set_ylabel('COP')
        ax2.set_title('Coefficient of Performance (COP)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # (3) Power consumption
    ax3 = axes[1, 0]
    if len(simulation_results) > 0:
        ax3.plot(simulation_results['timestamp'], simulation_results['power_normal'], 
                 'b-', label='Compressor Power (Full)', linewidth=2)
        ax3.plot(simulation_results['timestamp'], simulation_results['power_part_load'], 
                 'c-', label='Compressor Power (80%)', linewidth=2)
        ax3.set_ylabel('Power (kW)')
        ax3.set_title('Compressor Power Consumption')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # (4) Heat transfer rates
    ax4 = axes[1, 1]
    if len(simulation_results) > 0:
        ax4.plot(simulation_results['timestamp'], simulation_results['heat_output_normal'], 
                 'orange', label='Heat Output (Full)', linewidth=2)
        ax4.plot(simulation_results['timestamp'], simulation_results['heat_input_normal'], 
                 'purple', label='Heat Input (Full)', linewidth=2)
        ax4.set_ylabel('Heat Transfer Rate (kW)')
        ax4.set_title('Heat Transfer Rates')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    plt.tight_layout()
    plt.savefig('heat_pump_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Extra: scatter relationships (COP vs T_source, Power vs T_source)
    if len(simulation_results) > 0:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # COP vs Source Temperature
        scatter1 = ax1.scatter(simulation_results['T_source'], simulation_results['COP_normal'],
                               c=simulation_results['power_normal'], alpha=0.6, cmap='viridis')
        ax1.set_xlabel('Source Temperature (°C)')
        ax1.set_ylabel('COP')
        ax1.set_title('COP vs Source Temperature (Full Load)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Compressor Power (kW)')
        
        # Power vs Source Temperature
        scatter2 = ax2.scatter(simulation_results['T_source'], simulation_results['power_normal'],
                               c=simulation_results['COP_normal'], alpha=0.6, cmap='plasma')
        ax2.set_xlabel('Source Temperature (°C)')
        ax2.set_ylabel('Compressor Power (kW)')
        ax2.set_title('Power vs Source Temperature (Full Load)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='COP')
        
        plt.tight_layout()
        plt.savefig('heat_pump_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def run_off_design_analysis(heat_pump):
    """
    Evaluate several off-design scenarios (different source/sink temps and load).
    Returns a tidy DataFrame and prints a text summary.
    """
    print("\n" + "="*50)
    print("OFF-DESIGN PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Define a small set of representative cases
    scenarios = [
        {"name": "Base case",     "T_source": 40, "T_sink": 40, "load": 1.0},
        {"name": "Cold source",   "T_source": 35, "T_sink": 40, "load": 1.0},
        {"name": "Warm sink",     "T_source": 40, "T_sink": 45, "load": 1.0},
        {"name": "Part load",     "T_source": 40, "T_sink": 40, "load": 0.8},
        {"name": "Extreme cold",  "T_source": 30, "T_sink": 40, "load": 1.0},
    ]
    
    results = []
    
    for scenario in scenarios:
        # Try TESPy off-design first
        result = heat_pump.set_off_design_conditions(
            T_source_in=scenario["T_source"],
            T_sink_in=scenario["T_sink"],
            load_factor=scenario["load"]
        )
        
        if result:
            results.append({
                'Scenario': scenario["name"],
                'T_source': scenario["T_source"],
                'T_sink': scenario["T_sink"],
                'Load_factor': scenario["load"],
                'COP': result['COP'],
                'Compressor_Power': result['compressor_power'],
                'Heat_Output': result['heat_output'],
                'Heat_Input': result['heat_input']
            })
        else:
            # Analytical backup if TESPy fails to converge
            T_source_k = scenario["T_source"] + 273.15
            T_sink_k = scenario["T_sink"] + 273.15
            cop_carnot = T_sink_k / (T_sink_k - T_source_k) if T_sink_k > T_source_k else 0
            cop_actual = cop_carnot * 0.55 * scenario["load"]
            heat_output = 1000 * scenario["load"]
            power = heat_output / cop_actual if cop_actual > 0 else 0
            
            results.append({
                'Scenario': scenario["name"] + " (Analytical)",
                'T_source': scenario["T_source"],
                'T_sink': scenario["T_sink"],
                'Load_factor': scenario["load"],
                'COP': cop_actual,
                'Compressor_Power': power,
                'Heat_Output': heat_output,
                'Heat_Input': heat_output - power
            })
    
    # Print table to console
    results_df = pd.DataFrame(results)
    print("\nOff-Design Performance Summary:")
    print("="*80)
    print(f"{'Scenario':<20} {'T_source':<10} {'T_sink':<10} {'Load':<8} {'COP':<8} {'Power':<12} {'Heat_Out':<12}")
    print("-"*80)
    for _, row in results_df.iterrows():
        print(f"{row['Scenario']:<20} {row['T_source']:<10.1f} {row['T_sink']:<10.1f} {row['Load_factor']:<8.1f} "
              f"{row['COP']:<8.3f} {row['Compressor_Power']:<12.2f} {row['Heat_Output']:<12.2f}")
    
    return results_df

def main():
    """Entry point: load data, build model, solve design, run analyses, save outputs."""
    print("HEAT PUMP SYSTEM MODELING AND ANALYSIS")
    print("="*60)
    
    # Path to the Excel workbook with 'source' and 'sink' sheets
    excel_path = r"C:\Users\Saeed\OneDrive - Aalto University\Desktop\ai training\HP_case_data (2).xlsx"
    df = load_real_data(excel_path)
    
    # Build and configure the TESPy heat pump
    heat_pump = HeatPumpModel()
    
    # Solve the design point (with simplified fallback)
    design_success = heat_pump.set_design_conditions()
    
    if design_success:
        # Scenario sweep at off-design
        off_design_results = run_off_design_analysis(heat_pump)
        
        # Time-series off-design evaluation
        simulation_results = simulate_time_series(heat_pump, df)
        
        # Create plots
        create_performance_plots(df, simulation_results)
        
        # Save CSV outputs
        simulation_results.to_csv('heat_pump_time_series_results.csv', index=False)
        off_design_results.to_csv('heat_pump_off_design_results.csv', index=False)
        df.to_csv('source_temperature_data.csv', index=False)
        
        print(f"\n✓ Results saved to CSV files")
        print(f"✓ Visualizations saved as PNG files")
        
        # Quick summary stats
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        if len(simulation_results) > 0:
            print(f"Time Series Analysis:")
            print(f"  Average COP (Full Load): {simulation_results['COP_normal'].mean():.3f}")
            print(f"  Average COP (Part Load): {simulation_results['COP_part_load'].mean():.3f}")
            print(f"  Average Compressor Power: {simulation_results['power_normal'].mean():.2f} kW")
            print(f"  Average Heat Output: {simulation_results['heat_output_normal'].mean():.2f} kW")
            print(f"  Performance Range - COP: {simulation_results['COP_normal'].min():.2f} - {simulation_results['COP_normal'].max():.2f}")
    
    else:
        # If design never converged, proceed with analytical-only results
        print("Heat pump model configuration failed. Using analytical approach only.")
        simulation_results = create_analytical_results(df)
        create_performance_plots(df, simulation_results)
        simulation_results.to_csv('heat_pump_analytical_results.csv', index=False)

if __name__ == "__main__":
    main()
