import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm
import io
import base64
from PIL import Image

st.set_page_config(layout="wide", page_title="Heat Equation with Laser Ablation")

st.title("1D Heat Conduction Equation with Laser Ablation")
st.markdown("""
This app solves the 1D heat equation with laser ablation as a heat source:

$$\\frac{\\partial T}{\\partial t} = \\frac{k}{\\rho c_p}\\frac{\\partial^2 T}{\\partial x^2} + \\frac{I_0 e^{-\\alpha x}}{\\rho c_p}$$

Adjust the parameters below to see how they affect the temperature distribution.
""")

# Create a sidebar for input parameters
st.sidebar.header("Material Properties")
k = st.sidebar.number_input("Thermal Conductivity k (W/m·K)", value=20.0)
rho = st.sidebar.number_input("Density ρ (kg/m³)", value=8000.0)
c_p = st.sidebar.number_input("Specific Heat Capacity c_p (J/kg·K)", value=500.0)
diffusivity = k / (rho * c_p)

st.sidebar.header("Laser Properties")
I_0 = st.sidebar.number_input("Laser Intensity I₀ (W/m²)", value=1e6)
alpha = st.sidebar.number_input("Absorption Coefficient α (1/m)", value=1000.0)
laser_on = st.sidebar.checkbox("Laser Active", True)

st.sidebar.header("Initial & Boundary Conditions")
initial_temp = st.sidebar.number_input("Initial Temperature (K)", value=300.0)
bc_left_type = st.sidebar.selectbox("Left Boundary Type", ["Dirichlet", "Neumann"])
bc_left_value = st.sidebar.number_input(f"Left {'Temperature (K)' if bc_left_type == 'Dirichlet' else 'Heat Flux (W/m²)'}", value=300.0 if bc_left_type == 'Dirichlet' else 0.0)

bc_right_type = st.sidebar.selectbox("Right Boundary Type", ["Dirichlet", "Neumann"])
bc_right_value = st.sidebar.number_input(f"Right {'Temperature (K)' if bc_right_type == 'Dirichlet' else 'Heat Flux (W/m²)'}", value=300.0 if bc_right_type == 'Dirichlet' else 0.0)

st.sidebar.header("Discretization Parameters")
length = st.sidebar.number_input("Domain Length (m)", value=0.01)
nx = st.sidebar.number_input("Number of Spatial Points", value=50)
simulation_time = st.sidebar.number_input("Total Simulation Time (s)", value=1.0)
dt_input = st.sidebar.number_input("Time Step (s)", value=0.001, format="%.5f")
num_time_steps = int(simulation_time / dt_input)

# Calculate stability criterion for explicit method
dx = length / (nx - 1)
stability_dt = 0.5 * dx**2 / diffusivity
stability_status = "✅ Stable" if dt_input <= stability_dt else "❌ Unstable"
stability_color = "green" if dt_input <= stability_dt else "red"

st.sidebar.markdown(f"<p style='color:{stability_color};'>Stability status: {stability_status}</p>", unsafe_allow_html=True)
st.sidebar.markdown(f"Max stable time step: {stability_dt:.6f} s")

# Define t_values outside the button click block
# Initialize the spatial grid and solution arrays
x = np.linspace(0, length, nx)
T = np.ones(nx) * initial_temp
T_history = np.zeros((num_time_steps + 1, nx))
T_history[0] = T.copy()
t_values = np.linspace(0, simulation_time, num_time_steps + 1)

def laser_source(x, I0, alpha):
    """Calculate the laser heat source term."""
    if laser_on:
        return I0 * np.exp(-alpha * x) / (rho * c_p)
    else:
        return np.zeros_like(x)

def solve_explicit_step(T, dx, dt, diffusivity, source):
    """Solve one time step using explicit finite difference."""
    T_new = T.copy()
    
    # Interior points
    for i in range(1, nx-1):
        T_new[i] = T[i] + dt * (diffusivity * (T[i+1] - 2*T[i] + T[i-1]) / dx**2 + source[i])
    
    # Boundary conditions
    if bc_left_type == "Dirichlet":
        T_new[0] = bc_left_value
    else:  # Neumann
        T_new[0] = T_new[1] + bc_left_value * dx / k
        
    if bc_right_type == "Dirichlet":
        T_new[-1] = bc_right_value
    else:  # Neumann
        T_new[-1] = T_new[-2] - bc_right_value * dx / k
        
    return T_new

# Set up the main area for plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("Temperature Distribution Along X-axis")
    fig1, ax1 = plt.subplots(figsize=(8, 5))

# Progress bar and stopping condition
progress_bar = st.progress(0)
simulation_running = st.empty()
stop_button = st.button("Stop Simulation")

simulation_running.text("Click 'Run Simulation' to start")

# Run simulation when button is clicked
if st.button("Run Simulation"):
    # Reset arrays if running a new simulation
    T = np.ones(nx) * initial_temp
    T_history = np.zeros((num_time_steps + 1, nx))
    T_history[0] = T.copy()
    
    # Run the simulation
    simulation_running.text("Simulation running...")
    
    # Temperature plot placeholder
    temp_plot = st.empty()
    
    # Update plots in real-time
    for n in range(num_time_steps):
        if stop_button:
            simulation_running.text("Simulation stopped by user")
            break
            
        source = laser_source(x, I_0, alpha)
        
        T = solve_explicit_step(T, dx, dt_input, diffusivity, source)
            
        T_history[n+1] = T.copy()
        
        # Update progress
        progress = (n + 1) / num_time_steps
        progress_bar.progress(progress)
        
        # Update temperature plot
        if n % max(1, num_time_steps // 100) == 0:  # Update every ~1% of steps
            ax1.clear()
            ax1.plot(x * 1000, T, 'r-', linewidth=2)  # Convert to mm for display
            ax1.set_xlabel('Position (mm)')
            ax1.set_ylabel('Temperature (K)')
            ax1.set_title(f'Temperature Distribution at t = {(n+1)*dt_input:.4f} s')
            ax1.grid(True)
            
            # Display updated plots
            temp_plot.pyplot(fig1)
            
            time.sleep(0.01)  # Small delay to show animation effect
    
    # Final update
    simulation_running.text("Simulation completed!")
    progress_bar.progress(1.0)
    
    # Final plots
    ax1.clear()
    ax1.plot(x * 1000, T, 'r-', linewidth=2)  # Convert to mm for display
    ax1.set_xlabel('Position (mm)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title(f'Final Temperature Distribution at t = {simulation_time:.4f} s')
    ax1.grid(True)
    
    temp_plot.pyplot(fig1)
    
    # Display export option
    st.subheader("Export Results")
    
    # Function to create a download link for CSV
    def get_csv_download_link(data, filename):
        csv = io.StringIO()
        np.savetxt(csv, data, delimiter=',')
        csv_str = csv.getvalue()
        b64 = base64.b64encode(csv_str.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
        return href
    
    # Function to create a download link for the figure
    def get_figure_download_link(fig, filename):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">Download {filename}</a>'
        return href
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(get_csv_download_link(T_history, "temperature_history.csv"), unsafe_allow_html=True)
    with col2:
        st.markdown(get_figure_download_link(fig1, "temperature_distribution.png"), unsafe_allow_html=True)

# Display equations and explanation
st.markdown("""
## Mathematical Model

The 1D heat conduction equation with laser ablation is:

$$\\frac{\\partial T}{\\partial t} = \\frac{k}{\\rho c_p}\\frac{\\partial^2 T}{\\partial x^2} + \\frac{I_0 e^{-\\alpha x}}{\\rho c_p}$$

Where:
- $T$ is temperature (K)
- $t$ is time (s)
- $k$ is thermal conductivity (W/m·K)
- $\\rho$ is density (kg/m³)
- $c_p$ is specific heat capacity (J/kg·K)
- $\\frac{k}{\\rho c_p}$ is thermal diffusivity (m²/s)
- $I_0$ is laser intensity (W/m²)
- $\\alpha$ is absorption coefficient (1/m)
- $x$ is position (m)
""")

# Add a contour plot for temperature vs. time and space at the end of the simulation
with st.container():
    st.subheader("Temperature Contour Plot (Time vs. Space)")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    X, Y = np.meshgrid(x * 1000, t_values)  # Convert to mm for display
    contour = ax4.contourf(X, Y, T_history, 50, cmap=cm.hot)
    ax4.set_xlabel('Position (mm)')
    ax4.set_ylabel('Time (s)')
    ax4.set_title('Temperature Evolution')
    fig4.colorbar(contour, ax=ax4, label='Temperature (K)')
    st.pyplot(fig4)

# Add a footer with the user's name
st.markdown("---")
st.markdown("**Developed by Tanmay Jain**")