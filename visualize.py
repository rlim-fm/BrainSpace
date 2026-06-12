import numpy as np
import plotly.graph_objects as go

class Viusualizer:
    def __init__(self, data):
        self.data = data

    def plot2d(self, filepath=None):
        fig = go.Figure(data=self.data)

# 1. Build the grid and evaluate the function
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = Y**2 - X**2  # Analytic function: Hyperbolic Paraboloid

# 2. Create the 3D Surface figure
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Cividis')])

# 3. Fine-tune layout, titles, and scene parameters
fig.update_layout(
    title='Interactive Analytic Surface: $z = y^2 - x^2$',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ),
    autosize=False,
    width=800,
    height=800
)

# 4. Render the plot in your browser or notebook
fig.show()
