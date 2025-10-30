**🔥 Heat Pump Modeling Tool** 
A Python-based heat pump modeling and analysis tool built using TESPy (Thermal Engineering Systems in Python). The tool simulates design and off-design operation of a heat pump, evaluates system performance under varying conditions, and visualizes results through time-series and performance plots.

**🧩 Features** 
Object-oriented heat pump model with modular design. Design and off-design simulations using TESPy.

**✅ Calculation of:** Coefficient of Performance (COP) Compressor power consumption Heat transfer rates in condenser and evaporator Comprehensive visualization of system performance and parameter correlations.

**⚙️ Requirements** 
Python Libraries pip install tespy pandas numpy matplotlib
Dependencies TESPy ≥ 0.5.0 pandas numpy matplotlib

**🧠 Model Description** 
1. Design Condition: Compressor efficiency: 0.85 Heat source: 40→10°C, 1000 kW Heat sink: 40→90°C, 1012 kW
2. Off-Design Scenarios: Part-load operation (e.g., 80% load) Reduced source temperature (up to -5°C deviation)
   
**📊 Outputs File**
Description heat_pump_time_series_results.csv Hourly performance data (COP, power, heat rates) heat_pump_off_design_results.csv Results for multiple off-design scenarios heat_pump_performance_analysis.png Time-series plots of temperature, COP, and power

**📈 Example Visualizations**
COP vs Source Temperature Compressor Power vs Source Temperature Time-series performance of heat source and heat sink
