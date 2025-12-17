#!/usr/bin/env python3
"""
Room Impulse Response Generator

Generates impulse responses from a PNG floor plan using FDTD acoustic simulation.
The output can be convolved with any audio to simulate the room's acoustics.

Color conventions (RGB):
  - Black (0,0,0): Wall / solid boundary
  - White (255,255,255): Air / empty space
  
  Sources (impulse positions, marked by colored pixels):
  - Red (255,0,0): Source 1
  - Green (0,255,0): Source 2  
  - Blue (0,0,255): Source 3
  - Yellow (255,255,0): Source 4
  - Orange (255,128,0): Source 5
  
  Microphones (recording positions, marked by colored pixels):
  - Cyan (0,255,255): Microphone 1
  - Magenta (255,0,255): Microphone 2
  - Purple (128,0,255): Microphone 3

Usage:
  python app.py room.png --output impulse_response.wav --duration 2.0
  
  Or with a config file:
  python app.py room.png --config simulation.json

Example config.json:
{
    "scale": 0.1,           // meters per pixel
    "f_max": 1000,          // max frequency (Hz)
    "duration": 2.0,        // simulation duration (seconds)
    "courant": 0.2,         // Courant number
    "output_sr": 44100      // Output sample rate
}

To apply the impulse response to audio:
  from scipy.signal import convolve
  output = convolve(dry_audio, impulse_response)
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from fdtd import (
    FDTDGrid,
    Microphone,
    PML,
    RickerWaveletSource,
    Simulation,
    SimulationParams,
)
from ard import decompose_simple, extract_interfaces, Rectangle, RectangleInterface


# =============================================================================
# Color definitions
# =============================================================================

# Source colors (name -> RGB tuple)
SOURCE_COLORS = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'orange': (255, 128, 0),
}

# Microphone colors (name -> RGB tuple)
MIC_COLORS = {
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'purple': (128, 0, 255),
}

# Special colors
WALL_COLOR = (0, 0, 0)
AIR_COLOR = (255, 255, 255)


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """Euclidean distance between two RGB colors."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def identify_color(rgb: Tuple[int, int, int], threshold: float = 30.0) -> Optional[str]:
    """Identify what a color represents.
    
    Returns:
        One of: 'wall', 'air', 'source_<name>', 'mic_<name>', or None for unknown
    """
    # Check for exact or near matches
    if color_distance(rgb, WALL_COLOR) < threshold:
        return 'wall'
    if color_distance(rgb, AIR_COLOR) < threshold:
        return 'air'
    
    # Check source colors
    for name, color in SOURCE_COLORS.items():
        if color_distance(rgb, color) < threshold:
            return f'source_{name}'
    
    # Check microphone colors
    for name, color in MIC_COLORS.items():
        if color_distance(rgb, color) < threshold:
            return f'mic_{name}'
    
    # Gray shades are treated as partial walls (for absorption)
    if rgb[0] == rgb[1] == rgb[2]:
        return 'absorbing'
    
    return None


# =============================================================================
# PNG Parsing
# =============================================================================

@dataclass
class RoomGeometry:
    """Parsed room geometry from PNG."""
    occupancy: np.ndarray  # True = air, False = wall
    absorption: np.ndarray  # 0-1 absorption coefficient per cell
    sources: Dict[str, Tuple[int, int]]  # source_name -> (x, y) pixel coords
    microphones: Dict[str, Tuple[int, int]]  # mic_name -> (x, y) pixel coords
    width: int  # pixels
    height: int  # pixels


def parse_room_png(png_path: str) -> RoomGeometry:
    """Parse a PNG file to extract room geometry.
    
    Args:
        png_path: Path to PNG file
        
    Returns:
        RoomGeometry with occupancy grid, sources, and microphones
    """
    from PIL import Image
    
    img = Image.open(png_path).convert('RGB')
    pixels = np.array(img)
    
    height, width = pixels.shape[:2]
    
    # Initialize outputs
    occupancy = np.zeros((height, width), dtype=bool)
    absorption = np.zeros((height, width), dtype=float)
    sources: Dict[str, Tuple[int, int]] = {}
    microphones: Dict[str, Tuple[int, int]] = {}
    
    # Parse each pixel
    for y in range(height):
        for x in range(width):
            # Convert numpy uint8 to Python int for proper comparison
            rgb = tuple(int(c) for c in pixels[y, x])
            color_type = identify_color(rgb)
            
            if color_type == 'air':
                occupancy[y, x] = True
                absorption[y, x] = 0.0
            elif color_type == 'wall':
                occupancy[y, x] = False
                absorption[y, x] = 1.0
            elif color_type == 'absorbing':
                # Gray = partial absorption, lighter = less absorbing
                occupancy[y, x] = True
                absorption[y, x] = 1.0 - (rgb[0] / 255.0)
            elif color_type is not None and color_type.startswith('source_'):
                # Source position - treat as air
                occupancy[y, x] = True
                source_name = color_type.replace('source_', '')
                sources[source_name] = (x, y)
            elif color_type is not None and color_type.startswith('mic_'):
                # Microphone position - treat as air
                occupancy[y, x] = True
                mic_name = color_type.replace('mic_', '')
                microphones[mic_name] = (x, y)
            else:
                # Unknown color - treat as air
                occupancy[y, x] = True
    
    # Flip y-axis so origin is at bottom-left (matches FDTD convention)
    occupancy = np.flipud(occupancy)
    absorption = np.flipud(absorption)
    
    # Update source/mic coordinates for flipped y
    sources = {name: (x, height - 1 - y) for name, (x, y) in sources.items()}
    microphones = {name: (x, height - 1 - y) for name, (x, y) in microphones.items()}
    
    print(f"Parsed room: {width}x{height} pixels")
    print(f"  Air cells: {np.sum(occupancy)}")
    print(f"  Wall cells: {np.sum(~occupancy)}")
    print(f"  Sources: {sources}")
    print(f"  Microphones: {microphones}")
    
    return RoomGeometry(
        occupancy=occupancy,
        absorption=absorption,
        sources=sources,
        microphones=microphones,
        width=width,
        height=height,
    )


# =============================================================================
# Simulation Setup
# =============================================================================

def create_simulation(
    geometry: RoomGeometry,
    scale: float,
    f_max: float,
    courant: float,
    amplitude: float = 0.01,
) -> Tuple[Simulation, List[FDTDGrid], List[Microphone]]:
    """Create FDTD simulation from room geometry using decomposition.
    
    Decomposes the room into rectangular regions and creates FDTD grids for each,
    connected by interfaces. Uses impulse sources (Ricker wavelets) so the output
    is an impulse response that can be convolved with any audio.
    
    Args:
        geometry: Parsed room geometry
        scale: Meters per pixel
        f_max: Maximum frequency (Hz)
        courant: Courant number
        amplitude: Source amplitude scaling
        
    Returns:
        Tuple of (simulation, list of grids, microphones)
    """
    # Create simulation
    sim = Simulation(f_max=f_max, courant=courant)
    params = sim.params
    
    # Physical dimensions
    room_width = geometry.width * scale
    room_height = geometry.height * scale
    
    print(f"\nSimulation setup:")
    print(f"  Room size: {room_width:.2f} x {room_height:.2f} m")
    print(f"  Grid spacing: {params.h:.4f} m")
    print(f"  Time step: {params.dt:.6f} s")
    print(f"  Sample rate: {1/params.dt:.0f} Hz")
    
    # Decompose the room into rectangles
    rectangles = decompose_simple(geometry.occupancy, random_seed=42)
    rect_interfaces = extract_interfaces(rectangles)
    
    print(f"\nRoom decomposition: {len(rectangles)} regions, {len(rect_interfaces)} interfaces")
    
    # Create FDTD grid for each rectangle
    grids: List[FDTDGrid] = []
    for i, rect in enumerate(rectangles):
        # Physical size of this rectangle
        rect_width = rect.width * scale
        rect_height = rect.height * scale
        
        grid = FDTDGrid(x_size=rect_width, y_size=rect_height, params=params)
        
        # Store rectangle info for coordinate mapping
        grid.rect = rect  # type: ignore
        grid.origin_x = rect.x * scale  # type: ignore
        grid.origin_y = rect.y * scale  # type: ignore
        
        grids.append(grid)
        sim.add_simulation_component(grid)
        
        print(f"  Region {i}: ({rect.x}, {rect.y}) size {rect.width}x{rect.height} px "
              f"-> {grid.xs}x{grid.ys} cells")
    
    # Create interfaces between adjacent grids
    for iface in rect_interfaces:
        grid_a = grids[iface.rect_a]
        grid_b = grids[iface.rect_b]
        rect_a = rectangles[iface.rect_a]
        rect_b = rectangles[iface.rect_b]
        
        # Calculate the interface region in grid cell coordinates
        # The interface spans 'length' pixels starting at 'start' along the edge
        
        # Convert pixel positions to grid cell positions
        h = params.h
        
        if iface.edge_a in ('top', 'bottom'):
            # Horizontal interface - convert x coordinates
            start_px_a = iface.start
            end_px_a = start_px_a + iface.length
            
            start_cells_a = int(start_px_a * scale / h)
            end_cells_a = int(end_px_a * scale / h)
            
            abs_start = rect_a.x + iface.start
            start_px_b = abs_start - rect_b.x
            end_px_b = start_px_b + iface.length
            
            start_cells_b = int(start_px_b * scale / h)
            end_cells_b = int(end_px_b * scale / h)
            
            # Ensure same width by using minimum
            width_a = end_cells_a - start_cells_a
            width_b = end_cells_b - start_cells_b
            min_width = min(width_a, width_b)
            if min_width <= 0:
                continue
            end_cells_a = start_cells_a + min_width
            end_cells_b = start_cells_b + min_width
            
            region_a = (start_cells_a, end_cells_a)
            region_b = (start_cells_b, end_cells_b)
        else:
            # Vertical interface - convert y coordinates
            start_px_a = iface.start
            end_px_a = start_px_a + iface.length
            
            start_cells_a = int(start_px_a * scale / h)
            end_cells_a = int(end_px_a * scale / h)
            
            abs_start = rect_a.y + iface.start
            start_px_b = abs_start - rect_b.y
            end_px_b = start_px_b + iface.length
            
            start_cells_b = int(start_px_b * scale / h)
            end_cells_b = int(end_px_b * scale / h)
            
            # Ensure same height by using minimum
            height_a = end_cells_a - start_cells_a
            height_b = end_cells_b - start_cells_b
            min_height = min(height_a, height_b)
            if min_height <= 0:
                continue
            end_cells_a = start_cells_a + min_height
            end_cells_b = start_cells_b + min_height
            
            region_a = (start_cells_a, end_cells_a)
            region_b = (start_cells_b, end_cells_b)
        
        try:
            sim.add_interface(iface.edge_a, grid_a, iface.edge_b, grid_b,
                            region_a=region_a, region_b=region_b)
        except ValueError as e:
            print(f"  Warning: Could not create interface between regions "
                  f"{iface.rect_a} and {iface.rect_b}: {e}")
    
    # Helper to find which grid contains a point
    def find_grid_for_point(px: int, py: int) -> Optional[Tuple[FDTDGrid, float, float]]:
        """Find the grid containing pixel (px, py) and return local coords."""
        for grid in grids:
            rect = grid.rect  # type: ignore
            if rect.contains(px, py):
                # Convert to local physical coords within this grid
                local_x = (px - rect.x + 0.5) * scale
                local_y = (py - rect.y + 0.5) * scale
                return grid, local_x, local_y
        return None
    
    # Add impulse sources (Ricker wavelets) to appropriate grids
    # This generates an impulse response that can be convolved with any audio
    for color_name, (px, py) in geometry.sources.items():
        result = find_grid_for_point(px, py)
        if result is None:
            print(f"  Warning: Source '{color_name}' at ({px}, {py}) is not in any air region")
            continue
        
        grid, local_x, local_y = result
        
        # Use Ricker wavelet - a band-limited impulse with zero DC component
        source = RickerWaveletSource(local_x, local_y, f_peak=f_max / 2.0, amplitude=amplitude)
        grid.add_source(source)
        
        global_x = grid.origin_x + local_x  # type: ignore
        global_y = grid.origin_y + local_y  # type: ignore
        print(f"  Source '{color_name}' at ({global_x:.2f}, {global_y:.2f}) m (impulse)")
    
    # Add microphones to appropriate grids
    microphones: List[Microphone] = []
    for mic_name, (px, py) in geometry.microphones.items():
        result = find_grid_for_point(px, py)
        if result is None:
            print(f"  Warning: Microphone '{mic_name}' at ({px}, {py}) is not in any air region")
            continue
        
        grid, local_x, local_y = result
        mic = Microphone(local_x, local_y, name=mic_name)
        grid.add_microphone(mic)
        microphones.append(mic)
        global_x = grid.origin_x + local_x  # type: ignore
        global_y = grid.origin_y + local_y  # type: ignore
        print(f"  Microphone '{mic_name}' at ({global_x:.2f}, {global_y:.2f}) m")
    
    return sim, grids, microphones


def run_simulation(
    sim: Simulation,
    grids: List[FDTDGrid],
    microphones: List[Microphone],
    duration: float,
    geometry: RoomGeometry,
    scale: float,
    visualize: bool = False,
    visualize_every: int = 100,
) -> None:
    """Run the simulation.
    
    Args:
        sim: Simulation object
        grids: List of FDTD grids
        microphones: List of microphones
        duration: Simulation duration in seconds
        geometry: Room geometry for visualization
        scale: Meters per pixel
        visualize: Whether to show visualization
        visualize_every: Steps between visualization updates
    """
    params = sim.params
    total_steps = int(duration / params.dt)
    
    print(f"\nRunning simulation:")
    print(f"  Duration: {duration:.2f} s")
    print(f"  Total steps: {total_steps}")
    print(f"  Regions: {len(grids)}")
    
    if visualize:
        import matplotlib
        matplotlib.use('TkAgg')  # Interactive backend for visualization
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        fig, ax = plt.subplots(figsize=(8, 6))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        
        room_width = geometry.width * scale
        room_height = geometry.height * scale
        
        # Create composite pressure field for visualization
        def get_composite_field():
            """Combine all grid pressure fields into one image."""
            # Create output array matching geometry resolution
            h = params.h
            out_height = int(room_height / h)
            out_width = int(room_width / h)
            composite = np.zeros((out_height, out_width))
            
            for grid in grids:
                rect = grid.rect  # type: ignore
                # Map grid cells to composite array
                x_start = int(rect.x * scale / h)
                y_start = int(rect.y * scale / h)
                x_end = min(x_start + grid.xs, out_width)
                y_end = min(y_start + grid.ys, out_height)
                
                # Copy grid data
                gx_end = x_end - x_start
                gy_end = y_end - y_start
                composite[y_start:y_end, x_start:x_end] = grid.p_1[:gy_end, :gx_end]
            
            return composite
        
        composite = get_composite_field()
        im = ax.imshow(composite, origin='lower',
                       extent=[0, room_width, 0, room_height],
                       cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        
        # Overlay walls
        wall_mask = ~geometry.occupancy
        wall_rgba = np.zeros((*wall_mask.shape, 4))
        wall_rgba[wall_mask] = [0.3, 0.3, 0.3, 1.0]  # Dark gray for walls
        ax.imshow(wall_rgba, origin='lower', extent=[0, room_width, 0, room_height])
        
        fig.colorbar(im, cax=cax, label='Pressure')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        
        # Plot microphone positions (in global coords)
        for mic in microphones:
            # Find which grid this mic is in
            for grid in grids:
                if mic in grid.microphones:
                    global_x = grid.origin_x + mic.x  # type: ignore
                    global_y = grid.origin_y + mic.y  # type: ignore
                    ax.plot(global_x, global_y, 'ko', markersize=8)
                    ax.annotate(mic.name, (global_x, global_y), textcoords="offset points",
                               xytext=(5, 5), fontsize=8)
                    break
        
        def animate(frame):
            for _ in range(visualize_every):
                sim.step()
            
            t = sim.step_count * params.dt
            ax.set_title(f'Step {sim.step_count} (t = {t:.4f} s)')
            
            composite = get_composite_field()
            im.set_data(composite)
            
            # Auto-scale colorbar
            vmax = max(0.01, np.max(np.abs(composite)))
            im.set_clim(-vmax, vmax)
            
            return [im]
        
        num_frames = total_steps // visualize_every
        anim = animation.FuncAnimation(fig, animate, frames=num_frames,
                                       interval=50, blit=False)
        plt.tight_layout()
        plt.show()
    else:
        # Run without visualization
        for step in range(total_steps):
            sim.step()
            
            if step % 1000 == 0:
                t = step * params.dt
                max_p = max(np.max(np.abs(grid.p_1)) for grid in grids)
                print(f"  Step {step}/{total_steps} (t = {t:.3f} s), max |p| = {max_p:.4f}")
    
    print(f"  Simulation complete!")


def save_recordings(
    microphones: List[Microphone],
    output_path: str,
    output_sr: int = 44100,
    normalize: bool = True,
) -> None:
    """Save microphone recordings as WAV files.
    
    Args:
        microphones: List of microphones with recordings
        output_path: Output file path (or prefix for multiple mics)
        output_sr: Output sample rate
        normalize: Whether to normalize audio
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = '.wav'
    
    for i, mic in enumerate(microphones):
        if len(microphones) == 1:
            filename = f"{base}{ext}"
        else:
            filename = f"{base}_{mic.name}{ext}"
        
        mic.save_wav(filename, target_sr=output_sr, normalize=normalize)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate impulse response from a PNG floor plan.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('room_png', help='Path to room PNG file')
    parser.add_argument('-o', '--output', default='output/impulse_response.wav',
                       help='Output WAV file path (impulse response)')
    parser.add_argument('-c', '--config', help='JSON config file path')
    parser.add_argument('--scale', type=float, default=0.1,
                       help='Meters per pixel (default: 0.1)')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Simulation duration in seconds (default: 2.0)')
    parser.add_argument('--f-max', type=float, default=1000,
                       help='Maximum frequency in Hz (default: 1000)')
    parser.add_argument('--courant', type=float, default=0.2,
                       help='Courant number (default: 0.2)')
    parser.add_argument('--amplitude', type=float, default=0.01,
                       help='Source amplitude (default: 0.01)')
    parser.add_argument('--output-sr', type=int, default=44100,
                       help='Output sample rate (default: 44100)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show real-time visualization')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    
    # Merge CLI args with config (CLI takes precedence)
    scale = args.scale if args.scale != 0.1 or 'scale' not in config else config['scale']
    duration = args.duration if args.duration != 2.0 or 'duration' not in config else config['duration']
    f_max = args.f_max if args.f_max != 1000 or 'f_max' not in config else config['f_max']
    courant = args.courant if args.courant != 0.2 or 'courant' not in config else config['courant']
    amplitude = args.amplitude if args.amplitude != 0.01 or 'amplitude' not in config else config['amplitude']
    output_sr = args.output_sr if args.output_sr != 44100 or 'output_sr' not in config else config['output_sr']
    
    # Parse room geometry
    geometry = parse_room_png(args.room_png)
    
    # Check that we have sources and microphones
    if not geometry.sources:
        print("\nError: No sources found in PNG!")
        print("Add red (255,0,0), green (0,255,0), or blue (0,0,255) pixels to mark source positions.")
        return
    
    if not geometry.microphones:
        print("\nError: No microphones found in PNG!")
        print("Add cyan (0,255,255) or magenta (255,0,255) pixels to mark microphone positions.")
        return
    
    # Create simulation with impulse sources
    sim, grids, microphones = create_simulation(
        geometry=geometry,
        scale=scale,
        f_max=f_max,
        courant=courant,
        amplitude=amplitude,
    )
    
    run_simulation(
        sim=sim,
        grids=grids,
        microphones=microphones,
        duration=duration,
        geometry=geometry,
        scale=scale,
        visualize=args.visualize,
    )
    
    # Save recordings
    save_recordings(microphones, args.output, output_sr=output_sr)
    
    print(f"\nDone! Recordings saved to {args.output}")


if __name__ == '__main__':
    main()
