# 1D Heat Conduction Equation with Laser Ablation

This Streamlit application simulates the 1D heat conduction equation with laser ablation as a heat source. The app allows users to adjust various parameters to observe how they affect the temperature distribution over time and space.

## Features
- Interactive parameter adjustment using Streamlit's sidebar.
- Real-time simulation of temperature distribution along the x-axis.
- Contour plot showing temperature evolution over time and space.
- Export options for downloading simulation results and plots.

## Parameters
- **Material Properties**
  - Thermal Conductivity (k)
  - Density (ρ)
  - Specific Heat Capacity (c_p)
- **Laser Properties**
  - Laser Intensity (I₀)
  - Absorption Coefficient (α)
- **Initial & Boundary Conditions**
  - Initial Temperature
  - Left and Right Boundary Conditions (Dirichlet or Neumann)
- **Discretization Parameters**
  - Domain Length
  - Number of Spatial Points
  - Total Simulation Time
  - Time Step

## How to Run
1. Ensure you have Python and Streamlit installed.
2. Navigate to the project directory.
3. Run the app using the command:
   ```bash
   streamlit run app.py
   ```
4. Open the provided URL in your web browser to interact with the app.

## Developed by
Tanmay Jain 