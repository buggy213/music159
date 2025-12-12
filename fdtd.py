"""
# experiment: rectangle with PMLs on the edges + point source
# should yield something which looks like spherical pressure wave

def fdtd_rectangle_with_pml(x: float, y: float, f_max: float, courant: float, pml_coeff: float, viz: bool = True):
  h: float = fdtd_spacing(f_max, 8)

  # create computational grid (size of rectangle quantized by FDTD spacing)
  xs: int = int(x / h)
  ys: int = int(y / h)

  # big impulse (also completely nonphysical)
  # (units same as field)
  source_strength: float = 2.0
  source_x: int = xs // 2
  source_y: int = ys // 2

  p_0: np.ndarray = np.zeros((ys, xs))
  p_1: np.ndarray = np.zeros((ys, xs))

  # d = 3 stencil (6th order central difference discretization of 2nd derivative)
  stencil: np.ndarray = np.array([2, -27, 270, -490, 270, -27, 2]) / 180
  d = 3

  # d = 2 stencil (4th order central difference discretization of 1st derivative on staggered grid points)
  stencil_1: np.ndarray = np.array([1, -27, 27, -1]) / 24
  d_1 = 2

  # pml
  pml_thickness = 5
  top_p: np.ndarray = np.zeros((pml_thickness, xs))
  top_psi: np.ndarray = np.zeros_like(top_p)

  top_v_y: np.ndarray = np.zeros((pml_thickness + d_1, xs))
  top_v_x: np.ndarray = np.zeros((pml_thickness, xs + 1))

  # coupling buffers
  pml_to_rectangle_w = d
  pml_to_rectangle = np.zeros((pml_to_rectangle_w, xs))
  rectangle_to_pml_w = 2 * d_1 - 1
  rectangle_to_pml = np.zeros((rectangle_to_pml_w, xs))

  print(f"real width (w/o pml): {xs*h}")
  print(f"real height (w/o pml): {ys*h}")
  print(f"x cells: {xs}, y cells: {ys}")
  print(f'impulse source at ({source_x * h}, {source_y * h})')
  print(f'    strength: {source_strength}')

  # CFL condition - numerical speed > physical speed
  # (sqrt(2) * h) / (2 * dt) > c
  # dt < h / (c * sqrt(2))
  assert courant <= 1.0, "CFL condition violated"
  dt: float = courant * h / (c * np.sqrt(2))

  y_centered = np.linspace(pml_thickness + 0.5 - 1.0, 0.5, pml_thickness)
  y_staggered = np.linspace(pml_thickness, 0, pml_thickness + 1)

  # sigma = pml_coeff * (distance from edge)^2
  sigma_y_centered = pml_coeff * (y_centered * y_centered)
  sigma_y_centered = np.broadcast_to(sigma_y_centered[:, None], top_p.shape)

  sigma_y_staggered = pml_coeff * (y_staggered * y_staggered)
  sigma_y_staggered = np.pad(sigma_y_staggered, (0, d_1-1), mode='constant', constant_values=0)
  sigma_y_staggered = np.broadcast_to(sigma_y_staggered[:, None], top_v_y.shape)

  pml_factor_centered = np.exp(-sigma_y_centered * dt)
  pml_factor_staggered = np.exp(-sigma_y_staggered * dt)

  fig, ax = plt.subplots(figsize=(6,4))
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  ax.set_title(f'step 0 (t = 0)')
  ax.set_xlabel('x (m)')
  ax.set_ylabel('y (m)')
  im = ax.imshow(p_1, origin='lower', extent=[0, x, 0, y], cmap='RdBu_r', vmin=-1, vmax=1)
  fig.colorbar(im, cax)
  plt.show()


  step: int = 0
  def couple_pml_rectangle(rectangle_p, pml_p):
    nonlocal pml_to_rectangle, rectangle_to_pml, step

    pml_to_rectangle = pml_p[-pml_to_rectangle_w:, :]
    rectangle_to_pml = rectangle_p[:rectangle_to_pml_w, :]

    # if step == 900:
    #   print('== couple_pml_rectangle ==')
    #   print('pml pressure')
    #   print(pml_p)
    #   print('pml to rectangle')
    #   print(pml_to_rectangle)
    #   print('rectangle to pml')
    #   print(rectangle_to_pml)
    #   print()

  def pml_step(debug=False):
    nonlocal top_p, top_psi
    nonlocal top_v_x, top_v_y
    nonlocal rectangle_to_pml

    # if debug and 17 <= step <= 20:
    #   print(f'== pml_step {step} ==')
      
    #   plt.imshow(top_p)
    #   plt.title('pressure')
    #   plt.colorbar()
    #   plt.show()

    #   plt.imshow(top_psi)
    #   plt.title('psi')
    #   plt.colorbar()
    #   plt.show()

    #   plt.imshow(top_v_x)
    #   plt.title('v_x')
    #   plt.colorbar()
    #   plt.show()

    #   plt.imshow(top_v_y)
    #   plt.title('v_y')
    #   plt.colorbar()
    #   plt.show()
      
    combined_p = np.concat((top_p, rectangle_to_pml), axis=0)

    top_p_0_with_neumann_bc = np.pad(top_p, ((0,0), (d_1, d_1)), mode='reflect')
    # must use correlate, since stencil is no longer symmetric
    dp_dx = scipy.signal.correlate(
        top_p_0_with_neumann_bc,
        stencil_1[None, :] / h,
        mode='valid',
    )

    assert dp_dx.shape == top_v_x.shape

    combined_p_with_neumann_bc = np.pad(combined_p, ((d_1, 0), (0, 0)), mode='reflect')
    dp_dy = scipy.signal.correlate(
        combined_p_with_neumann_bc,
        stencil_1[:, None] / h,
        mode='valid'
    )

    assert dp_dy.shape == top_v_y.shape

    top_v_x += (1.0 / rho_0) * dp_dx * dt
    top_v_y += (1.0 / rho_0) * dp_dy * dt
    top_v_y *= pml_factor_staggered

    # note: this doesn't really make sense physically, but
    # the wave should be so strongly damped by this point that it should be
    # kinda ok-ish
    top_v_x_with_dirichlet_bc = np.pad(top_v_x, ((0, 0), (d_1-1, d_1-1)), mode='constant', constant_values=0)
    dvx_dx = scipy.signal.correlate(
        top_v_x_with_dirichlet_bc,
        stencil_1[None, :] / h,
        mode='valid'
    )

    assert dvx_dx.shape == top_p.shape

    top_v_y_with_dirichlet_bc = np.pad(top_v_y, ((d_1-1, 0), (0, 0)), mode='constant', constant_values=0)
    dvy_dy = scipy.signal.correlate(
        top_v_y_with_dirichlet_bc,
        stencil_1[:, None] / h,
        mode='valid'
    )

    assert dvy_dy.shape == top_p.shape

    top_psi += K_0 * sigma_y_centered * dvx_dx * dt
    top_p += (K_0 * (dvx_dx + dvy_dy) + top_psi) * dt
    top_p *= pml_factor_centered

  def run_step():
    nonlocal p_0, p_1, step, pml_to_rectangle

    # mode='reflect' to enforce von Neumann boundary condition
    d2x = scipy.ndimage.convolve(p_1, stencil, mode='reflect', axes=(1,))

    combined_p = np.concat((pml_to_rectangle, p_1), axis=0)
    d2y = scipy.ndimage.convolve(combined_p, stencil, mode='reflect', axes=(0,))
    d2y = d2y[pml_to_rectangle_w:, :]

    # leapfrog integration
    p_0 = 2.0 * p_1 - p_0 + ((c * c * dt * dt) / (h * h)) * (d2x + d2y)
    p_0, p_1 = p_1, p_0

    # inject impulse
    if step == 0:
      p_1[source_y, source_x] += source_strength

    step += 1

  def combined_step():
    couple_pml_rectangle(p_1, top_p)
    run_step()
    pml_step()

  visualize_every: int = 100
  def visualize_pressure(_i):
    for _ in range(visualize_every):
      combined_step()

    ax.set_title(f'step {step} (t = {step*dt})')
    im.set_data(p_1)

  steps = 5000
  if viz:
    anim = animation.FuncAnimation(fig, visualize_pressure, frames=steps//visualize_every, interval=500)
    return anim
  else:
    for _ in range(steps):
      combined_step()
    return p_1

# %matplotlib inline
# _ = fdtd_rectangle_with_pml(1, 1, 1000, 0.2, 5, viz=False)

%matplotlib notebook
anim = fdtd_rectangle_with_pml(10, 10, 1000, 0.2, 800)
HTML(anim.to_html5_video())
"""

import math
import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile
from typing import Optional, Tuple, List, Dict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable

def fdtd_spacing(f_max: float, points_per_wavelength: int) -> float:
    """Calculate FDTD grid spacing from maximum frequency.
    
    Args:
        f_max: Maximum frequency (Hz)
        points_per_wavelength: Number of grid points per wavelength
        
    Returns:
        Grid spacing h (m)
    """
    # c = f * lambda, so lambda = c / f
    # h = lambda / points_per_wavelength
    # We need c, but we'll compute it from rho_0 and K_0
    rho_0 = 1.225
    K_0 = 144120
    c = math.sqrt(K_0 / rho_0)
    wavelength = c / f_max
    return wavelength / points_per_wavelength


class Source:
    """Base class for acoustic sources."""
    
    def __init__(self, x: float, y: float, amplitude: float = 1.0):
        """Initialize source.
        
        Args:
            x: X position (m)
            y: Y position (m)
            amplitude: Source amplitude
        """
        self.x = x
        self.y = y
        self.amplitude = amplitude
    
    def get_value(self, t: float, dt: float) -> float:
        """Get source value at time t.
        
        Args:
            t: Current time (s)
            dt: Time step (s)
            
        Returns:
            Source value at time t
        """
        raise NotImplementedError


# Note: impulse source and gaussian pulse source are non-zero mean.
# Neumann BC's don't constrain DC mode, which our integration scheme can cause to grow linearly
# I will think of a way to deal with this later, I suppose 

class ImpulseSource(Source):
    """Single impulse at t=0."""
    
    def get_value(self, t: float, dt: float) -> float:
        # Fire only on first timestep
        if t < dt:
            return self.amplitude
        return 0.0


class GaussianPulseSource(Source):
    """Band-limited Gaussian pulse (smooth impulse).
    
    Centered at t=t0 with width controlled by f_max.
    """
    
    def __init__(self, x: float, y: float, f_max: float, amplitude: float = 1.0, t0: float = None):
        """Initialize Gaussian pulse source.
        
        Args:
            x: X position (m)
            y: Y position (m)
            f_max: Maximum frequency content (Hz) - controls pulse width
            amplitude: Source amplitude
            t0: Pulse center time (s). If None, defaults to 4/f_max for smooth onset.
        """
        super().__init__(x, y, amplitude)
        self.f_max = f_max
        # Gaussian width parameter: sigma = 1/(pi*f_max) gives -6dB at f_max
        self.sigma = 1.0 / (np.pi * f_max)
        # Default center time: 4 sigma ensures smooth onset (starts near zero)
        self.t0 = t0 if t0 is not None else 4.0 * self.sigma
    
    def get_value(self, t: float, dt: float) -> float:
        tau = (t - self.t0) / self.sigma
        return self.amplitude * np.exp(-0.5 * tau * tau)


class SinusoidalSource(Source):
    """Continuous sinusoidal source."""
    
    def __init__(self, x: float, y: float, frequency: float, amplitude: float = 1.0, phase: float = 0.0):
        """Initialize sinusoidal source.
        
        Args:
            x: X position (m)
            y: Y position (m)
            frequency: Oscillation frequency (Hz)
            amplitude: Source amplitude
            phase: Initial phase (radians)
        """
        super().__init__(x, y, amplitude)
        self.frequency = frequency
        self.phase = phase
    
    def get_value(self, t: float, dt: float) -> float:
        return self.amplitude * np.sin(2.0 * np.pi * self.frequency * t + self.phase)


class GaussianModulatedSineSource(Source):
    """Gaussian-modulated sinusoid (tone burst).
    
    A sinusoid windowed by a Gaussian envelope - good for 
    band-limited excitation at a specific center frequency.
    """
    
    def __init__(self, x: float, y: float, frequency: float, n_cycles: float = 4.0,
                 amplitude: float = 1.0, t0: float = None):
        """Initialize Gaussian-modulated sine source.
        
        Args:
            x: X position (m)
            y: Y position (m)
            frequency: Center frequency (Hz)
            n_cycles: Number of cycles within the Gaussian envelope (controls bandwidth)
            amplitude: Source amplitude
            t0: Envelope center time (s). If None, computed for smooth onset.
        """
        super().__init__(x, y, amplitude)
        self.frequency = frequency
        self.n_cycles = n_cycles
        # Gaussian width: n_cycles periods at the center frequency
        self.sigma = n_cycles / (2.0 * np.pi * frequency)
        # Default center time: 4 sigma for smooth onset
        self.t0 = t0 if t0 is not None else 4.0 * self.sigma
    
    def get_value(self, t: float, dt: float) -> float:
        tau = (t - self.t0) / self.sigma
        envelope = np.exp(-0.5 * tau * tau)
        return self.amplitude * envelope * np.sin(2.0 * np.pi * self.frequency * t)


class RickerWaveletSource(Source):
    """Ricker wavelet (Mexican hat) source.
    
    Second derivative of a Gaussian - commonly used in seismic/acoustic 
    simulations. Has zero DC component and well-defined bandwidth.
    """
    
    def __init__(self, x: float, y: float, f_peak: float, amplitude: float = 1.0, t0: float = None):
        """Initialize Ricker wavelet source.
        
        Args:
            x: X position (m)
            y: Y position (m)
            f_peak: Peak frequency (Hz)
            amplitude: Source amplitude
            t0: Wavelet center time (s). If None, computed for smooth onset.
        """
        super().__init__(x, y, amplitude)
        self.f_peak = f_peak
        # For Ricker wavelet, sigma relates to peak frequency
        self.sigma = np.sqrt(2.0) / (np.pi * f_peak)
        # Default center time: 4 sigma for smooth onset
        self.t0 = t0 if t0 is not None else 4.0 * self.sigma
    
    def get_value(self, t: float, dt: float) -> float:
        tau = (t - self.t0) / self.sigma
        tau_sq = tau * tau
        # Ricker wavelet: (1 - tau^2) * exp(-tau^2/2)
        return self.amplitude * (1.0 - tau_sq) * np.exp(-0.5 * tau_sq)


class SoftOnsetSineSource(Source):
    """Sinusoid with smooth (tanh) onset to avoid startup transients."""
    
    def __init__(self, x: float, y: float, frequency: float, amplitude: float = 1.0,
                 onset_cycles: float = 2.0, phase: float = 0.0):
        """Initialize soft-onset sinusoidal source.
        
        Args:
            x: X position (m)
            y: Y position (m)
            frequency: Oscillation frequency (Hz)
            amplitude: Source amplitude
            onset_cycles: Number of cycles for amplitude to ramp up
            phase: Initial phase (radians)
        """
        super().__init__(x, y, amplitude)
        self.frequency = frequency
        self.phase = phase
        # Time constant for tanh ramp
        self.onset_time = onset_cycles / frequency
    
    def get_value(self, t: float, dt: float) -> float:
        # Smooth ramp from 0 to 1
        ramp = 0.5 * (1.0 + np.tanh(4.0 * (t / self.onset_time - 0.5)))
        return self.amplitude * ramp * np.sin(2.0 * np.pi * self.frequency * t + self.phase)


class Microphone:
    """Virtual microphone to record pressure at a point for auralization."""
    
    def __init__(self, x: float, y: float, name: str = "mic"):
        """Initialize microphone.
        
        Args:
            x: X position (m)
            y: Y position (m)
            name: Identifier for the microphone
        """
        self.x = x
        self.y = y
        self.name = name
        self.recordings: List[float] = []
        self._grid: Optional['FDTDGrid'] = None
        self._params: Optional['SimulationParams'] = None
    
    def attach_to_grid(self, grid: 'FDTDGrid', params: 'SimulationParams'):
        """Attach microphone to a grid for recording.
        
        Args:
            grid: The FDTDGrid to record from
            params: Simulation parameters
        """
        self._grid = grid
        self._params = params
        self.recordings = []
    
    def record(self):
        """Record the current pressure at the microphone position.
        
        Uses bilinear interpolation for sub-cell accuracy.
        """
        if self._grid is None or self._params is None:
            raise RuntimeError("Microphone not attached to a grid. Call attach_to_grid first.")
        
        h = self._params.h
        
        # Get continuous grid coordinates
        fx = self.x / h
        fy = self.y / h
        
        # Integer indices for surrounding cells
        ix0 = int(np.floor(fx))
        iy0 = int(np.floor(fy))
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        
        # Clamp to valid grid range
        ix0 = max(0, min(ix0, self._grid.xs - 1))
        ix1 = max(0, min(ix1, self._grid.xs - 1))
        iy0 = max(0, min(iy0, self._grid.ys - 1))
        iy1 = max(0, min(iy1, self._grid.ys - 1))
        
        # Bilinear interpolation weights
        wx = fx - np.floor(fx)
        wy = fy - np.floor(fy)
        
        # Bilinear interpolation
        p = self._grid.p_1
        value = (
            p[iy0, ix0] * (1 - wx) * (1 - wy) +
            p[iy0, ix1] * wx * (1 - wy) +
            p[iy1, ix0] * (1 - wx) * wy +
            p[iy1, ix1] * wx * wy
        )
        
        self.recordings.append(value)
    
    def get_native_sample_rate(self) -> float:
        """Get the native sample rate of the simulation (1/dt)."""
        if self._params is None:
            raise RuntimeError("Microphone not attached to a grid.")
        return 1.0 / self._params.dt
    
    def get_audio(self, target_sr: int = 44100, normalize: bool = True) -> Tuple[np.ndarray, int]:
        """Get recorded signal, resampled to target sample rate.
        
        Args:
            target_sr: Target sample rate in Hz (default: 44100)
            normalize: Whether to normalize audio to [-1, 1] range
            
        Returns:
            Tuple of (audio signal as numpy array, sample rate)
        """
        if len(self.recordings) == 0:
            raise RuntimeError("No recordings. Run the simulation first.")
        
        signal = np.array(self.recordings, dtype=np.float64)
        
        # Native sample rate from simulation
        native_sr = self.get_native_sample_rate()
        
        # Resample to target sample rate
        # Number of samples at target rate
        duration = len(signal) / native_sr
        num_samples = int(duration * target_sr)
        
        if num_samples < 1:
            num_samples = 1
        
        # Use scipy's resample for high-quality resampling
        audio = scipy.signal.resample(signal, num_samples)
        
        # Normalize if requested
        if normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
        
        return audio, target_sr
    
    def save_wav(self, filename: str, target_sr: int = 44100, 
                 normalize: bool = True):
        """Save recording as a WAV file.
        
        Args:
            filename: Output filename (should end in .wav)
            target_sr: Target sample rate in Hz
            normalize: Whether to normalize audio
            bit_depth: Bit depth (16 or 32)
        """
        audio, sr = self.get_audio(target_sr=target_sr, normalize=normalize)
        
        # Convert to appropriate integer format
        audio_int = (audio * 32767).astype(np.int16)
        
        wavfile.write(filename, sr, audio_int)
        
        duration = len(audio) / sr
        print(f"Saved {duration:.3f}s of audio to {filename} ({sr}Hz, 16-bit)")
    
    def get_raw_recording(self) -> Tuple[np.ndarray, float]:
        """Get the raw recording at native sample rate.
        
        Returns:
            Tuple of (raw signal, native sample rate)
        """
        return np.array(self.recordings), self.get_native_sample_rate()


class SimulationParams:
    """Stores all physical and numerical parameters for the simulation."""
    
    def __init__(self, f_max: float, courant: float, rho_0: float = 1.225, K_0: float = 144120):
        """Initialize simulation parameters.
        
        Args:
            f_max: Maximum frequency (Hz)
            courant: Courant number (must be <= 1.0)
            rho_0: Density (kg/m³)
            K_0: Bulk modulus (Pa)
        """
        self.rho_0 = rho_0
        self.K_0 = K_0
        self.c = np.sqrt(K_0 / rho_0)
        self.f_max = f_max
        self.courant = courant
        
        # Grid spacing
        self.h = fdtd_spacing(f_max, 8)
        
        # Time step from CFL condition
        assert courant <= 1.0, "CFL condition violated"
        self.dt = courant * self.h / (self.c * np.sqrt(2))
        
        # Stencils
        # d = 3 stencil (6th order central difference discretization of 2nd derivative)
        self.stencil = np.array([2, -27, 270, -490, 270, -27, 2]) / 180
        self.d = 3
        
        # d_1 = 2 stencil (4th order central difference discretization of 1st derivative)
        self.stencil_1 = np.array([1, -27, 27, -1]) / 24
        self.d_1 = 2
        
        # Coupling buffer widths
        self.grid_halo = self.d
        self.pml_halo = 2 * self.d_1 - 1

class Interface:
    """Handles bidirectional data transfer between components (FDTDGrid-FDTDGrid or FDTDGrid-PML)."""
    
    def __init__(self, edge_a: str, component_a: 'SimulationComponent', 
                 edge_b: str, component_b: 'SimulationComponent', 
                 params: SimulationParams, 
                 region_a: Optional[Tuple[int, int]] = None,
                 region_b: Optional[Tuple[int, int]] = None):
        """Initialize interface.
        
        Args:
            edge_a: Edge name relative to component_a ('top', 'bottom', 'left', 'right')
            component_a: First component (FDTDGrid or PML)
            edge_b: Edge name relative to component_b ('top', 'bottom', 'left', 'right')
            component_b: Second component (FDTDGrid or PML)
            params: Simulation parameters
            region_a: Optional (start, end) tuple for partial edge coverage on component_a. If None, spans full edge.
            region_b: Optional (start, end) tuple for partial edge coverage on component_b. If None, spans full edge.
        """
        self.edge_a = edge_a
        self.edge_b = edge_b
        self.component_a = component_a
        self.component_b = component_b
        self.params = params
        self.region_a = region_a
        self.region_b = region_b
        
        # Get edge dimensions from components
        dim_a = component_a.get_edge_dimension(edge_a)
        dim_b = component_b.get_edge_dimension(edge_b)
        
        # Validate regions are in bounds
        if region_a is not None:
            start_a, end_a = region_a
            if start_a < 0 or end_a > dim_a or start_a >= end_a:
                raise ValueError(f"Invalid region_a {region_a} for component_a edge {edge_a} (dimension {dim_a})")
        
        if region_b is not None:
            start_b, end_b = region_b
            if start_b < 0 or end_b > dim_b or start_b >= end_b:
                raise ValueError(f"Invalid region_b {region_b} for component_b edge {edge_b} (dimension {dim_b})")
        
        # Determine buffer dimensions based on edge orientation
        if edge_a in ['top', 'bottom']:
            # Interface along x-axis (horizontal)
            if region_a is None:
                width_a = dim_a
            else:
                width_a = region_a[1] - region_a[0]
            
            if region_b is None:
                width_b = dim_b
            else:
                width_b = region_b[1] - region_b[0]
            
            # Both edges should have the same width for a valid interface
            if width_a != width_b:
                raise ValueError(f"Interface width mismatch: component_a edge {edge_a} width {width_a} != "
                               f"component_b edge {edge_b} width {width_b}")
            
            
            self.b_to_a_buffer = np.zeros((component_a.halo_size(), width_a))
            self.a_to_b_buffer = np.zeros((component_b.halo_size(), width_b))
            
        else:  # left or right - interface along y-axis (vertical)
            if region_a is None:
                height_a = dim_a
            else:
                height_a = region_a[1] - region_a[0]
            
            if region_b is None:
                height_b = dim_b
            else:
                height_b = region_b[1] - region_b[0]
            
            # Both edges should have the same height for a valid interface
            if height_a != height_b:
                raise ValueError(f"Interface height mismatch: component_a edge {edge_a} height {height_a} != "
                               f"component_b edge {edge_b} height {height_b}")
            
            
            self.b_to_a_buffer = np.zeros((height_a, component_a.halo_size()))
            self.a_to_b_buffer = np.zeros((height_b, component_b.halo_size()))
    
    def get_region_a(self) -> Optional[Tuple[int, int]]:
        """Return the region this interface covers on component_a."""
        return self.region_a
    
    def get_region_b(self) -> Optional[Tuple[int, int]]:
        """Return the region this interface covers on component_b."""
        return self.region_b
    
    def update(self):
        """Perform bidirectional data transfer."""
        # Transfer from component_a to buffer, then to component_b
        self.component_a.transfer_to_interface(self)
        self.component_b.transfer_from_interface(self)
        
        # Transfer from component_b to buffer, then to component_a
        self.component_b.transfer_to_interface(self)
        self.component_a.transfer_from_interface(self)

class SimulationComponent:
    """Base class for all simulation components."""
    
    def step(self):
        """Perform one time step."""
        raise NotImplementedError

    def before_transfer(self):
        """Do stuff before transfer to/from interface 
        (i.e. set default boundary conditions)
        """
        raise NotImplementedError
    
    def transfer_to_interface(self, interface: 'Interface'):
        """Transfer data to interface buffer."""
        raise NotImplementedError
    
    def transfer_from_interface(self, interface: 'Interface'):
        """Transfer data from interface buffer."""
        raise NotImplementedError
    
    def get_edge_dimension(self, edge: str) -> int:
        """Get the dimension (length) of an edge.
        
        Args:
            edge: Edge name ('top', 'bottom', 'left', 'right')
            
        Returns:
            Length of the edge in cells
        """
        raise NotImplementedError

    def halo_size(self) -> int:
        raise NotImplementedError

class Simulation:
    """Orchestrates the entire simulation."""
    
    def __init__(self, f_max: float, courant: float, rho_0: float = 1.225, K_0: float = 144120):
        """Initialize simulation.
        
        Args:
            f_max: Maximum frequency (Hz)
            courant: Courant number (must be <= 1.0)
            rho_0: Density (kg/m³)
            K_0: Bulk modulus (Pa)
        """
        self.params = SimulationParams(f_max, courant, rho_0, K_0)
        self.components: List['SimulationComponent'] = []
        self.interfaces: List[Interface] = []
        self.step_count = 0
    
    def add_simulation_component(self, component: 'SimulationComponent'):
        """Register a simulation component."""
        self.components.append(component)
    
    def add_interface(self, edge_a: str, component_a: 'SimulationComponent', 
                     edge_b: str, component_b: 'SimulationComponent',
                     region_a: Optional[Tuple[int, int]] = None,
                     region_b: Optional[Tuple[int, int]] = None):
        """Create and register interface between two components.
        
        Args:
            edge_a: Edge name relative to component_a ('top', 'bottom', 'left', 'right')
            component_a: First component (FDTDGrid or PML)
            edge_b: Edge name relative to component_b ('top', 'bottom', 'left', 'right')
            component_b: Second component (FDTDGrid or PML)
            region_a: Optional (start, end) tuple for partial edge coverage on component_a
            region_b: Optional (start, end) tuple for partial edge coverage on component_b. 
        """
        # Prevent PML-PML interfaces
        if isinstance(component_a, PML) and isinstance(component_b, PML):
            raise ValueError("PML-PML interfaces are not allowed. Interfaces must connect at least one FDTDGrid.")
        
        interface = Interface(edge_a, component_a, edge_b, component_b, self.params, region_a, region_b)
        self.interfaces.append(interface)
        
        # Register interface with components
        if isinstance(component_a, FDTDGrid):
            component_a.add_interface(edge_a, interface)
        if isinstance(component_b, FDTDGrid):
            component_b.add_interface(edge_b, interface)
        if isinstance(component_a, PML):
            component_a.interface = interface
        if isinstance(component_b, PML):
            component_b.interface = interface
    
    def step(self):
        """Perform one simulation step."""
        for component in self.components:
            component.before_transfer()

        # Transfer data through interfaces
        for interface in self.interfaces:
            interface.update()
        
        # Step all components
        for component in self.components:
            component.step()
        
        self.step_count += 1
    
    def run(self, steps: int):
        """Run simulation for specified number of steps.
        
        Args:
            steps: Number of time steps to run
        """
        for _ in range(steps):
            self.step()

class FDTDGrid(SimulationComponent):
    """Main computational grid for FDTD simulation."""
    
    def __init__(self, x_size: float, y_size: float, params: SimulationParams):
        """Initialize FDTD grid.
        
        Args:
            x_size: Physical width (m)
            y_size: Physical height (m)
            params: Simulation parameters
        """
        self.params = params
        self.xs = int(x_size / params.h)
        self.ys = int(y_size / params.h)
        
        # Pressure fields for leapfrog integration
        self.p_0 = np.zeros((self.ys, self.xs))
        self.p_1 = np.zeros((self.ys, self.xs))
        
        self.left_p = np.zeros((self.ys, params.grid_halo))
        self.right_p = np.zeros((self.ys, params.grid_halo))
        self.top_p = np.zeros((params.grid_halo, self.xs))
        self.bottom_p = np.zeros((params.grid_halo, self.xs))

        # Interfaces: dict mapping edge -> list of (region, interface) tuples
        self.interfaces: Dict[str, List[Tuple[Optional[Tuple[int, int]], Interface]]] = {
            'top': [], 'bottom': [], 'left': [], 'right': []
        }
        
        # Sources
        self.sources: List[Source] = []
        
        # Microphones
        self.microphones: List[Microphone] = []
        
        self.step_count = 0
    
    def get_edge_dimension(self, edge: str) -> int:
        """Get the dimension (length) of an edge.
        
        Args:
            edge: Edge name ('top', 'bottom', 'left', 'right')
            
        Returns:
            Length of the edge in cells
        """
        if edge in ['top', 'bottom']:
            return self.xs
        else:  # left or right
            return self.ys

    def before_transfer(self):
        # von Neumann boundary conditions
        # IMPORTANT: must use .copy() because [::-1] creates a view, and we don't want
        # transfer_from_interface to accidentally write into p_1
        self.top_p = self.p_1[:self.params.grid_halo, :][::-1, :].copy()
        self.bottom_p = self.p_1[-self.params.grid_halo:, :][::-1, :].copy()
        self.left_p = self.p_1[:, :self.params.grid_halo][:, ::-1].copy()
        self.right_p = self.p_1[:, -self.params.grid_halo:][:, ::-1].copy()
    
    def add_interface(self, edge: str, interface: Interface):
        """Register interface on specific edge."""
        # Determine which edge of this grid the interface is on
        if interface.component_a is self:
            region = interface.get_region_a()
        elif interface.component_b is self:
            region = interface.get_region_b()
        else:
            raise ValueError("Interface does not connect to this grid")
        
        self.interfaces[edge].append((region, interface))
    
    def add_source(self, source: Source):
        """Add a source to the grid.
        
        Args:
            source: Source object (ImpulseSource, GaussianPulseSource, etc.)
        """
        self.sources.append(source)
    
    def add_microphone(self, microphone: Microphone):
        """Add a microphone to the grid for recording.
        
        Args:
            microphone: Microphone object
        """
        microphone.attach_to_grid(self, self.params)
        self.microphones.append(microphone)
    
    def step(self):
        """Perform one FDTD time step."""
        params = self.params
        
        # Use explicit boundary arrays for all four sides
        # (boundary arrays contain either PML data or reflection BC from before_transfer)
        p_with_x_borders = np.concat((self.left_p, self.p_1, self.right_p), axis=1)
        d2x = scipy.signal.correlate(p_with_x_borders, params.stencil[None, :], mode='valid')
        
        p_with_y_borders = np.concat((self.top_p, self.p_1, self.bottom_p), axis=0)
        d2y = scipy.signal.correlate(p_with_y_borders, params.stencil[:, None], mode='valid')
        
        # Leapfrog integration
        c_sq_dt_sq_h_sq = (params.c * params.c * params.dt * params.dt) / (params.h * params.h)
        self.p_0 = 2.0 * self.p_1 - self.p_0 + c_sq_dt_sq_h_sq * (d2x + d2y)
        self.p_0, self.p_1 = self.p_1, self.p_0
        
        # Inject sources
        t = self.step_count * params.dt
        for source in self.sources:
            source_x = int(source.x / params.h)
            source_y = int(source.y / params.h)
            if 0 <= source_x < self.xs and 0 <= source_y < self.ys:
                self.p_1[source_y, source_x] += source.get_value(t, params.dt)
        
        # Record from microphones
        for mic in self.microphones:
            mic.record()
        
        self.step_count += 1
    
    def transfer_to_interface(self, interface: Interface):
        """Copy boundary data to interface buffer."""
        # Determine which edge of this grid the interface is on
        if interface.component_a is self:
            edge = interface.edge_a
            region = interface.region_a
            target = interface.a_to_b_buffer
            target_halo_size = interface.component_b.halo_size()
        elif interface.component_b is self:
            edge = interface.edge_b
            region = interface.region_b
            target = interface.b_to_a_buffer
            target_halo_size = interface.component_a.halo_size()
        else:
            raise ValueError("Interface does not connect to this grid")
        
        if edge == 'top':
            if region is None:
                start, end = 0, self.xs
            else:
                start, end = region
            np.copyto(target, self.p_1[:target_halo_size, start:end])
        
        elif edge == 'bottom':
            if region is None:
                start, end = 0, self.xs
            else:
                start, end = region
            np.copyto(target, self.p_1[-target_halo_size:, start:end])
        
        elif edge == 'left':
            if region is None:
                start, end = 0, self.ys
            else:
                start, end = region
            np.copyto(target, self.p_1[start:end, :target_halo_size])
        
        else:  # right
            if region is None:
                start, end = 0, self.ys
            else:
                start, end = region
            np.copyto(target, self.p_1[start:end, -target_halo_size:])
    
    def transfer_from_interface(self, interface: Interface):
        """Copy boundary data from interface buffer."""
        
        if self is interface.component_a:
            region = interface.region_a
            edge = interface.edge_a
            source = interface.b_to_a_buffer
        else:
            region = interface.region_b
            edge = interface.edge_b
            source = interface.a_to_b_buffer
        
        if region is None:
            start, end = 0, (self.xs if edge in ('top', 'bottom') else self.ys)
        else:
            start, end = region

        match edge:
            case 'top':
                np.copyto(self.top_p[:, start:end], source)
            case 'bottom':
                np.copyto(self.bottom_p[:, start:end], source)
            case 'left':
                np.copyto(self.left_p[start:end, :], source)
            case 'right':
                np.copyto(self.right_p[start:end, :], source)
    
    def halo_size(self) -> int:
        return self.params.grid_halo

class PML(SimulationComponent):
    """Perfectly Matched Layer absorbing boundary."""
    
    def __init__(
        self, 
        thickness: int, 
        width: int,
        direction: str, 
        pml_coeff: float, 
        params: SimulationParams, 
    ):
        """Initialize PML.
        
        Args:
            thickness: Number of cells in PML
            direction: Normal direction ('up', 'down', 'left', 'right'), PML absorbs waves incident from this direction
            width: width perpendicular to normal
            pml_coeff: PML damping coefficient
            params: Simulation parameters
        """
        self.thickness = thickness
        self.direction = direction
        self.width = width
        self.pml_coeff = pml_coeff
        self.params = params

        # canonical PML is facing down, use rotations in coupling code
        # to match real direction

        self.p = np.zeros((thickness, width))
        self.halo_p = np.zeros((params.pml_halo, width))
        self.psi = np.zeros_like(self.p)
        self.v_y = np.zeros((thickness + params.d_1, width))
        self.v_x = np.zeros((thickness, width + 1))
        
        y_centered = np.linspace(thickness + 0.5 - 1.0, 0.5, thickness)
        y_staggered = np.linspace(thickness, 0, thickness + 1)
        y_staggered_padding = (0, params.d_1-1)

        sigma_y_centered = pml_coeff * (y_centered * y_centered)
        self.sigma_centered = np.broadcast_to(sigma_y_centered[:, None], self.p.shape)
        
        sigma_y_staggered = pml_coeff * (y_staggered * y_staggered)
        sigma_y_staggered = np.pad(sigma_y_staggered, y_staggered_padding, mode='constant', constant_values=0)
        self.sigma_staggered = np.broadcast_to(sigma_y_staggered[:, None], self.v_y.shape)
        
        self.pml_factor_centered = np.exp(-self.sigma_centered * params.dt)
        self.pml_factor_staggered = np.exp(-self.sigma_staggered * params.dt)
    
    def get_edge_dimension(self, edge: str) -> int:
        """Get the dimension (length) of an edge.
        
        Args:
            edge: Edge name ('top', 'bottom', 'left', 'right')
            
        Returns:
            Length of the edge in cells
        """
        if edge in ['top', 'bottom']:
            if self.direction in ['up', 'down']:
                return self.width
            else:
                return self.thickness
        else:
            if self.direction in ['up', 'down']:
                return self.thickness
            else:
                return self.width
    

    def before_transfer(self):
        # there isn't a reasonable "default" boundary condition for PML,
        # so we don't need to do anything here
        pass

    def step(self):
        """Perform one PML time step."""
        params = self.params
        h = params.h
        dt = params.dt
        rho_0 = params.rho_0
        K_0 = params.K_0
        d_1 = params.d_1
        stencil = self.params.stencil_1
        
        combined_p = np.concatenate((self.p, self.halo_p), axis=0)

        padded_p = np.pad(self.p, ((0,0), (d_1,d_1)), mode='reflect')
        dp_dx = scipy.signal.correlate(
            padded_p,
            stencil[None, :] / h,
            mode='valid'
        )

        assert dp_dx.shape == self.v_x.shape

        combined_p_with_neumann_bc = np.pad(combined_p, ((d_1, 0), (0,0)), mode='reflect')
        dp_dy = scipy.signal.correlate(
            combined_p_with_neumann_bc,
            stencil[:, None] / h,
            mode='valid'
        )

        assert dp_dy.shape == self.v_y.shape

        self.v_x += (1.0 / rho_0) * dp_dx * dt
        self.v_y += (1.0 / rho_0) * dp_dy * dt
        self.v_y *= self.pml_factor_staggered

        padded_vx = np.pad(self.v_x, ((0,0), (d_1-1, d_1-1)), mode='constant', constant_values=0)
        dvx_dx = scipy.signal.correlate(
            padded_vx,
            stencil[None, :] / h,
            mode='valid'
        )
        
        assert dvx_dx.shape == self.p.shape
        
        padded_vy = np.pad(self.v_y, ((d_1-1,0), (0,0)), mode='constant', constant_values=0)
        dvy_dy = scipy.signal.correlate(
            padded_vy,
            stencil[:, None] / h,
            mode='valid'
        )

        assert dvy_dy.shape == self.p.shape

        self.psi += K_0 * self.sigma_centered * dvx_dx * dt
        self.p += (K_0 * (dvx_dx + dvy_dy) + self.psi) * dt
        self.p *= self.pml_factor_centered


        
    def transfer_to_interface(self, interface: Interface):
        """Copy boundary data to interface buffer."""
        # Determine which edge of this PML the interface is on
        if interface.component_a is self:
            dest = interface.a_to_b_buffer
            dest_halo_size = interface.component_b.halo_size()
        elif interface.component_b is self:
            dest = interface.b_to_a_buffer
            dest_halo_size = interface.component_a.halo_size()
        else:
            raise ValueError("Interface does not connect to this PML")
        
        # base[-1, :] is AT the PML-grid interface (low damping)
        base = self.p[-dest_halo_size:, :]

        if self.direction == 'up':
            # bottom_p[0, :] should be AT interface, so flip rows
            np.copyto(dest, np.flip(base, axis=0))
        elif self.direction == 'down':
            # top_p[-1, :] should be AT interface, matches base[-1, :]
            np.copyto(dest, base)
        elif self.direction == 'left':
            # pml_right: right_p[:, 0] should be AT interface
            # Transpose (3, ys) → (ys, 3), then flip columns so col 0 = base[-1, :]
            np.copyto(dest, base.T[:, ::-1])
        else:  # direction == 'right'
            # pml_left: left_p[:, -1] should be AT interface
            # Transpose (3, ys) → (ys, 3), col -1 = base[-1, :] ✓
            np.copyto(dest, base.T)
        
        
    def transfer_from_interface(self, interface: Interface):
        """Copy boundary data from interface buffer."""
        
        if interface.component_a is self:
            source = interface.b_to_a_buffer
        elif interface.component_b is self:
            source = interface.a_to_b_buffer
        else:
            raise ValueError("Interface does not connect to this PML")
        
        # halo_p[0, :] should be the grid cells closest to the PML interface
        
        if self.direction == 'up':
            # Grid sends p_1[-3:, :], where p_1[-1, :] is AT interface
            # Flip so halo_p[0, :] = source[-1, :]
            np.copyto(self.halo_p, np.flip(source, axis=0))
        elif self.direction == 'down':
            # Grid sends p_1[:3, :], where p_1[0, :] is AT interface
            # halo_p[0, :] = source[0, :] ✓
            np.copyto(self.halo_p, source)
        elif self.direction == 'left':
            # pml_right: Grid sends p_1[:, -3:] shape (ys, 3), p_1[:, -1] is AT interface
            # Need halo_p[0, :] = source[:, -1]
            # Transpose (ys, 3) → (3, ys), then flip rows so row 0 = old row -1
            np.copyto(self.halo_p, source.T[::-1, :])
        else:  # direction == 'right'
            # pml_left: Grid sends p_1[:, :3] shape (ys, 3), p_1[:, 0] is AT interface
            # Need halo_p[0, :] = source[:, 0]
            # Transpose (ys, 3) → (3, ys), row 0 = source[:, 0] ✓
            np.copyto(self.halo_p, source.T)

    def halo_size(self) -> int:
        return self.params.pml_halo

def fdtd_rectangle_with_pml(x: float, y: float, f_max: float, courant: float, 
                            pml_coeff: float, pml_thickness: int = 5, 
                            source_strength: float = 2.0, steps: int = 5000,
                            visualize_every: int = 100, viz: bool = True,
                            output_file: Optional[str] = None,
                            mic_positions: Optional[List[Tuple[float, float]]] = None,
                            audio_output_prefix: Optional[str] = None,
                            audio_sr: int = 44100):
    """Create and run FDTD simulation with PML boundaries and visualization.
    
    This function recreates the original notebook functionality using the new
    class-based API.
    
    Args:
        x: Physical width (m)
        y: Physical height (m)
        f_max: Maximum frequency (Hz)
        courant: Courant number (must be <= 1.0)
        pml_coeff: PML damping coefficient
        pml_thickness: PML thickness in cells (default: 5)
        source_strength: Source strength (default: 2.0)
        steps: Number of time steps (default: 5000)
        visualize_every: Update visualization every N steps (default: 100)
        viz: Whether to create visualization (default: True)
        output_file: Optional filename to save animation (e.g., 'animation.gif' or 'animation.mp4').
                    If None and viz=True, displays interactively. If None and viz=False, returns final field.
        mic_positions: Optional list of (x, y) tuples for microphone positions (in meters).
                      If provided, recordings will be saved as WAV files.
        audio_output_prefix: Prefix for audio output files (default: 'fdtd_audio').
                            Files will be named '{prefix}_mic{i}.wav'
        audio_sr: Sample rate for audio output (default: 44100 Hz)
        
    Returns:
        If viz=True and output_file is set: saves animation to file, returns microphones (if any)
        If viz=True and output_file is None: matplotlib animation object
        If viz=False: tuple of (final pressure field p_1, microphones list)
    """
    # Create simulation
    sim = Simulation(f_max=f_max, courant=courant)
    
    # Create FDTD grid
    grid = FDTDGrid(x_size=x, y_size=y, params=sim.params)
    
    # Add source at center
    source_x = x / 2.0
    source_y = y / 2.0
    # Use band-limited Gaussian pulse instead of impulse to avoid aliasing issues
    grid.add_source(RickerWaveletSource(source_x, source_y, f_peak=f_max / 2.0, amplitude=source_strength / 100.0))
    
    # Add microphones if specified
    microphones: List[Microphone] = []
    if mic_positions is not None:
        for i, (mx, my) in enumerate(mic_positions):
            mic = Microphone(mx, my, name=f"mic{i}")
            grid.add_microphone(mic)
            microphones.append(mic)
            print(f"Microphone {i} at ({mx:.2f}, {my:.2f}) m")
    
    # Add to simulation
    sim.add_simulation_component(grid)
    
    # Create PMLs for all 4 sides
    # Direction indicates where the wave is coming FROM (the PML absorbs waves incident from that direction)
    pml_top = PML(thickness=pml_thickness, width=grid.xs, direction='down', 
                  pml_coeff=pml_coeff, params=sim.params)
    pml_bottom = PML(thickness=pml_thickness, width=grid.xs, direction='up', 
                     pml_coeff=pml_coeff, params=sim.params)
    pml_left = PML(thickness=pml_thickness, width=grid.ys, direction='right', 
                   pml_coeff=pml_coeff, params=sim.params)
    pml_right = PML(thickness=pml_thickness, width=grid.ys, direction='left', 
                    pml_coeff=pml_coeff, params=sim.params)
    
    # Add PMLs to simulation
    # sim.add_simulation_component(pml_top)
    # sim.add_simulation_component(pml_bottom)
    # sim.add_simulation_component(pml_left)
    # sim.add_simulation_component(pml_right)
    
    # Create interfaces between grid and PMLs
    # sim.add_interface('top', grid, 'bottom', pml_top)
    # sim.add_interface('bottom', grid, 'top', pml_bottom)
    # sim.add_interface('left', grid, 'right', pml_left)
    # sim.add_interface('right', grid, 'left', pml_right)
    
    # Print info
    print(f"real width (w/o pml): {grid.xs * sim.params.h}")
    print(f"real height (w/o pml): {grid.ys * sim.params.h}")
    print(f"x cells: {grid.xs}, y cells: {grid.ys}")
    print(f'impulse source at ({source_x}, {source_y})')
    print(f'    strength: {source_strength}')
    
    if not viz:
        # Run without visualization
        sim.run(steps=steps)
        
        # Save audio from microphones
        if mic_positions is not None and len(microphones) > 0:
            prefix = audio_output_prefix or 'fdtd_audio'
            for i, mic in enumerate(microphones):
                filename = f"{prefix}_mic{i}.wav"
                mic.save_wav(filename, target_sr=audio_sr)
        
        return grid.p_1, microphones
    
    # Setup visualization
    # Use non-interactive backend if saving to file
    if output_file is not None:
        plt.ioff()  # Turn off interactive mode
        # Try to use a backend that supports file writing
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
        except:
            pass
    
    fig, ax = plt.subplots(figsize=(6, 4))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.set_title(f'step 0 (t = 0)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    # Initial plot - run one step first to inject source
    sim.step()
    im = ax.imshow(grid.p_1, origin='lower', 
                   extent=[0, x, 0, y], 
                   cmap='RdBu_r', vmin=-1, vmax=1)
    fig.colorbar(im, cax=cax)
    
    # Energy and mean pressure tracking
    h = sim.params.h
    dt = sim.params.dt
    c = sim.params.c
    
    def compute_energy_and_stats():
        """Compute total energy, mean pressure, and max pressure."""
        p = grid.p_1
        p_old = grid.p_0  # Previous timestep (after swap, p_0 holds n-1)
        
        # Time derivative: (p^n - p^{n-1}) / dt (one-sided since we don't have p^{n+1} yet)
        # Actually after the swap, p_1 is new, p_0 is old
        p_t = (p - p_old) / dt
        
        # Spatial gradients using central differences
        dp_dx = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * h)
        dp_dy = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * h)
        
        # Energy density: (1/2) * [p_t^2/c^2 + |∇p|^2]
        kinetic = 0.5 * (p_t ** 2) / (c ** 2)
        potential = 0.5 * (dp_dx ** 2 + dp_dy ** 2)
        
        # Total energy (sum over grid, multiply by cell area h^2)
        total_energy = np.sum(kinetic + potential) * h * h
        mean_pressure = np.mean(p)
        max_pressure = np.max(np.abs(p))
        
        return total_energy, mean_pressure, max_pressure
    
    # Track over time
    times = [dt]
    energies = []
    mean_pressures = []
    max_pressures = []
    
    e, m, mx = compute_energy_and_stats()
    energies.append(e)
    mean_pressures.append(m)
    max_pressures.append(mx)
    
    # Animation function
    step_count = [1]  # Start at 1 since we already ran one step
    
    def animate(frame):
        # Run multiple steps between frames
        for _ in range(visualize_every):
            sim.step()
            step_count[0] += 1
        
        # Track energy and stats
        e, m, mx = compute_energy_and_stats()
        times.append(step_count[0] * dt)
        energies.append(e)
        mean_pressures.append(m)
        max_pressures.append(mx)
        
        # Update plot
        ax.set_title(f'step {step_count[0]} (t = {step_count[0] * sim.params.dt:.4f})')
        im.set_data(grid.p_1)
        # Keep colorbar scale fixed (set at initialization)
        return [im]
    
    # Create animation (interval=200ms for slower playback)
    num_frames = steps // visualize_every
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                   interval=200, blit=False, repeat=True)
    
    plt.tight_layout()
    
    # Save to file or display
    if output_file is not None:
        print(f"Saving animation to {output_file}...")
        # Determine writer based on file extension
        if output_file.endswith('.gif'):
            try:
                writer = animation.PillowWriter(fps=5)  # Slower fps for GIF
                anim.save(output_file, writer=writer)
            except Exception as e:
                print(f"Error saving GIF: {e}")
                print("Trying alternative method...")
                writer = animation.ImageMagickWriter(fps=5)
                anim.save(output_file, writer=writer)
        elif output_file.endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=10)  # Slower fps
            anim.save(output_file, writer=writer)
        else:
            # Default to GIF
            output_file = output_file + '.gif'
            writer = animation.PillowWriter(fps=5)
            anim.save(output_file, writer=writer)
        print(f"Animation saved to {output_file}")
        plt.close(fig)
        
        # Plot energy and mean pressure
        fig2, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        axes[0].plot(times, energies, 'b-')
        axes[0].set_ylabel('Total Energy')
        axes[0].set_title('Energy and Pressure Statistics Over Time')
        axes[0].grid(True)
        
        axes[1].plot(times, mean_pressures, 'r-')
        axes[1].set_ylabel('Mean Pressure')
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1].grid(True)
        
        axes[2].plot(times, max_pressures, 'g-')
        axes[2].set_ylabel('Max |Pressure|')
        axes[2].set_xlabel('Time (s)')
        axes[2].grid(True)
        
        plt.tight_layout()
        stats_file = output_file.rsplit('.', 1)[0] + '_stats.png'
        plt.savefig(stats_file)
        print(f"Stats plot saved to {stats_file}")
        plt.close(fig2)
        
        # Save audio from microphones
        if mic_positions is not None and len(microphones) > 0:
            prefix = audio_output_prefix or 'fdtd_audio'
            for i, mic in enumerate(microphones):
                filename = f"{prefix}_mic{i}.wav"
                mic.save_wav(filename, target_sr=audio_sr)
        
        return microphones if microphones else None
    else:
        plt.show()  # Blocking show to keep window open
        return anim


# Example usage (same as notebook)
if __name__ == "__main__":
    # Same parameters as the original notebook
    # Save as GIF for terminal usage
    # pml_thickness=15 for better absorption (original was 5, which may cause energy issues)
    fdtd_rectangle_with_pml(x=10, y=10, f_max=1000, courant=0.2, pml_coeff=800,
                           pml_thickness=15, output_file='fdtd_animation.gif')
    # Keep animation in scope to prevent garbage collection
    try:
        input("Press Enter to exit...")  # Keep the window open
    except KeyboardInterrupt:
        pass