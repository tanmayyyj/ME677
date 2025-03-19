import streamlit as st
import numpy as np
import plotly.graph_objects as go
import io
import base64

st.set_page_config(layout="wide", page_title="Moving Heat Source Temperature Distribution")

st.title("Temperature Distribution for Moving Heat Source")
st.markdown("""
This app visualizes the temperature distribution around a moving heat source using the equation:

$$T - T_0 = \\frac{q}{2\\pi Kr} \\exp\\left(-\\frac{U_x(r + X)}{2D}\\right)$$

where:
- $T$: Temperature at a point
- $T_0$: Initial temperature
- $q$: Heat flux input
- $K$: Thermal conductivity
- $r$: Distance from heat source
- $U_x$: Velocity in x-direction
- $X$: Coordinate in moving system
- $D$: Thermal diffusivity
""")

# Sidebar inputs
st.sidebar.header("Material Properties")
T0 = st.sidebar.number_input("Initial Temperature T₀ (K)", value=300.0)
K = st.sidebar.number_input("Thermal Conductivity K (W/m·K)", value=20.0)
rho = st.sidebar.number_input("Density ρ (kg/m³)", value=8000.0)
cp = st.sidebar.number_input("Specific Heat cₚ (J/kg·K)", value=500.0)
D = K / (rho * cp)  # Thermal diffusivity

st.sidebar.header("Heat Source Properties")
q = st.sidebar.number_input("Heat Flux q (W)", value=1000.0)
Ux = st.sidebar.number_input("Velocity Uₓ (m/s)", value=0.1)
t = st.sidebar.number_input("Time t (s)", value=1.0)

st.sidebar.header("Plot Parameters")
x_min = st.sidebar.number_input("X min (m)", value=-0.1)
x_max = st.sidebar.number_input("X max (m)", value=0.1)
y_min = st.sidebar.number_input("Y min (m)", value=-0.1)
y_max = st.sidebar.number_input("Y max (m)", value=0.1)
n_points = st.sidebar.number_input("Grid resolution", value=50, min_value=10, max_value=200)

# Color scale selection
color_scales = {
    "Hot": "hot",
    "Viridis": "viridis",
    "Plasma": "plasma",
    "Magma": "magma",
    "Cool Warm": "coolwarm",
    "Rainbow": "rainbow",
    "Jet": "jet",
    "Earth": "earth",
    "Electric": "electric",
    "Thermal": "thermal"
}
selected_scale = st.sidebar.selectbox("Color Scale", list(color_scales.keys()), index=0)

def create_plot():
    try:
        # Create the grid
        x = np.linspace(x_min, x_max, n_points)
        y = np.linspace(y_min, y_max, n_points)
        X, Y = np.meshgrid(x, y)
        
        # Calculate r (distance from heat source)
        r = np.sqrt(X**2 + Y**2)
        
        # Calculate temperature distribution
        # Avoid division by zero by adding a small number to r
        r = np.where(r < 1e-10, 1e-10, r)
        T = T0 + (q / (2 * np.pi * K * r)) * np.exp(-Ux * (r + X) / (2 * D))
        
        # Create interactive 3D surface plot using Plotly
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=T, colorscale=color_scales[selected_scale])])
        
        # Update layout
        fig.update_layout(
            title=f'Temperature Distribution at t = {t:.2f} s',
            scene = dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Temperature (K)'
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting function: {str(e)}")
        return None

# Main plot area
if st.button("Generate Plot"):
    fig = create_plot()
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.subheader("Export Results")
        
        def get_figure_download_link(fig, filename):
            img_bytes = fig.to_image(format="png")
            img_str = base64.b64encode(img_bytes).decode()
            href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">Download {filename}</a>'
            return href
        
        st.markdown(get_figure_download_link(fig, "temperature_distribution.png"), unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("**Developed by Tanmay Jain**")