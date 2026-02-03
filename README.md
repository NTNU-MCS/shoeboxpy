# shoeboxpy

[![DOI](https://zenodo.org/badge/959946842.svg)](https://doi.org/10.5281/zenodo.17277252)

<div align="center">
    <img src="https://github.com/incebellipipo/shoeboxpy/blob/master/docs/assets/shoeboxpy.png?raw=true" alt="shoeboxpy" width="400"/>
</div>

## Installation

```bash
pip install git+https://github.com/incebellipipo/shoeboxpy.git
```

## Documentation

```bash
pip install -r requirements.docs.txt
```

To build the HTML documentation with version selectors, run:

```powershell
.\docs\make.bat multiversion
```

By default, documentation is generated for the `main` or `master` branch plus tags matching `v*`. Set `SMV_LATEST_VERSION` before building to rename a different branch to `latest`.

## Theory

The `shoeboxpy` package provides simulation models for vessels with 3 and 6 degrees of freedom (DOF). These models are based on rigid-body dynamics, including added mass, damping, Coriolis/centripetal effects, and restoring forces.

### 6-DOF Model

The 6-DOF model represents a rectangular "shoebox" vessel with the following states:

- Position and orientation in the inertial frame:
  $\eta = [x, y, z, \phi, \theta, \psi]$
- Velocities in the body frame:
  $\nu = [u, v, w, p, q, r]$

### 3-DOF Model

The 3-DOF model simplifies the dynamics to planar motion (surge, sway, yaw) with states:

- Position and orientation in the inertial frame:
  $\eta = [x, y, \psi]$
- Velocities in the body frame:
  $\nu = [u, v, r]$

### Dynamics

Dynamics for both models are governed by:

$$
\begin{aligned}
& \dot{\eta} = J(\eta)\nu\\
& (M_{RB} + M_A)\dot{\nu} + (C_{RB}(\nu) + C_A(\nu))\nu + D\nu = \tau + \tau_{\mathrm{ext}} + g_{\mathrm{restoring}}(\eta)
\end{aligned}
$$

where:

- $M_{RB}$ and $M_A$ are the rigid-body and added mass matrices.
- $C_{RB}(\nu)$ and $C_A(\nu)$ are the Coriolis/centripetal matrices.
- $D$ is the linear damping matrix.
- $\tau$ and $\tau_{\mathrm{ext}}$ are control and external forces/moments.
- $g_{\mathrm{restoring}}(\eta)$ represents restoring forces in roll and pitch.

Both models use a 4th-order Runge-Kutta method for numerical integration, allowing for accurate simulation of vessel dynamics under various forces and moments.

Read the [theory](./docs/theory.md) for more details.

## Example

Here is an example of how to use the `shoeboxpy` package to simulate a 6-DOF vessel:

```python
import numpy as np
from shoeboxpy.model6dof import Shoebox
from shoeboxpy.animate import animate_history

# Initialize the shoebox model
shoebox = Shoebox(
    L=1.0,  # Length (m)
    B=0.3,  # Width (m)
    T=0.03,  # Height (m)
    eta0=np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1]),  # Initial position and orientation
    nu0=np.zeros(6),  # Initial velocities
    GM_phi=0.2,  # Metacentric height in roll
    GM_theta=0.2,  # Metacentric height in pitch
)

# Simulate for 10 seconds with a time step of 0.01 seconds
dt = 0.01
eta_history = []
for _ in range(int(10 / dt)):
    shoebox.step(tau=np.array([1.0, 0.2, 0.0, 0.0, 0.0, 0.1]), dt=dt)  # Apply control forces
    eta_history.append(shoebox.get_states()[0])  # Store position and orientation

# Convert history to a NumPy array for analysis or visualization
eta_history = np.array(eta_history)

animate_history(eta_history, dt=dt, L=1.0, B=0.3, T=0.2)  # Animate the results
```
