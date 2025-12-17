#!/usr/bin/env python3
"""
Room Impulse Response Generator

Generates impulse responses from a PNG floor plan using FDTD acoustic simulation.
The output can be convolved with any audio to simulate the room's acoustics.

Color conventions (RGB):
  - Black (0,0,0): Fully reflective wall
  - Gray shades: Partially absorbing wall (darker = more absorbing)
  - White (255,255,255): Air / empty space
  - Red (255,0,0): Source position
  - Cyan (0,255,255): Microphone position

Usage:
  python app.py room.png --output impulse_response.wav --duration 2.0

To apply the impulse response to audio:
  from scipy.signal import convolve
  output = convolve(dry_audio, impulse_response)
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from fdtd import (
    FDTDGrid,
    Interface,
    Microphone,
    PML,
    RickerWaveletSource,
    Simulation,
    SimulationParams,
)
from ard import decompose_simple, extract_interfaces


# =============================================================================
# Color definitions
# =============================================================================

SOURCE_COLOR = (255, 0, 0)      # Red
MIC_COLOR = (0, 255, 255)       # Cyan
AIR_COLOR = (255, 255, 255)     # White


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """Euclidean distance between two RGB colors."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def identify_pixel(rgb: Tuple[int, int, int], threshold: float = 30.0) -> Tuple[str, float]:
    """Identify what a pixel represents.
    
    Returns:
        Tuple of (type, absorption) where:
        - type: 'air', 'source', 'mic', or 'wall'
        - absorption: 0.0 (fully reflective) to 1.0 (fully absorbing) for walls
    """
    if color_distance(rgb, SOURCE_COLOR) < threshold:
        return 'source', 0.0
    if color_distance(rgb, MIC_COLOR) < threshold:
        return 'mic', 0.0
    if color_distance(rgb, AIR_COLOR) < threshold:
        return 'air', 0.0
    
    # Everything else is a wall - brightness determines reflectivity
    # Black (0,0,0) = fully reflective (absorption=0)
    # White-ish gray = more absorbing
    brightness = (rgb[0] + rgb[1] + rgb[2]) / (3 * 255)
    # Invert: darker = less absorbing (more reflective)
    absorption = brightness
    return 'wall', absorption


# =============================================================================
# Room Geometry
# =============================================================================

@dataclass
class RoomGeometry:
    """Parsed room geometry from PNG."""
    occupancy: np.ndarray      # True = air, False = wall
    wall_absorption: np.ndarray  # Absorption coefficient for wall pixels (0-1)
    source_pos: Optional[Tuple[int, int]]  # (x, y) pixel coords
    mic_pos: Optional[Tuple[int, int]]     # (x, y) pixel coords
    width: int
    height: int


def parse_room_png(png_path: str) -> RoomGeometry:
    """Parse a PNG file to extract room geometry.
    
    Args:
        png_path: Path to PNG file
        
    Returns:
        RoomGeometry with occupancy grid, wall absorption, source and mic positions
    """
    from PIL import Image
    
    img = Image.open(png_path).convert('RGB')
    pixels = np.array(img)
    
    height, width = pixels.shape[:2]
    
    occupancy = np.zeros((height, width), dtype=bool)
    wall_absorption = np.zeros((height, width), dtype=float)
    source_pos: Optional[Tuple[int, int]] = None
    mic_pos: Optional[Tuple[int, int]] = None
    
    for y in range(height):
        for x in range(width):
            rgb = tuple(int(c) for c in pixels[y, x])
            pixel_type, absorption = identify_pixel(rgb)
            
            if pixel_type == 'air':
                occupancy[y, x] = True
            elif pixel_type == 'source':
                occupancy[y, x] = True
                source_pos = (x, y)
            elif pixel_type == 'mic':
                occupancy[y, x] = True
                mic_pos = (x, y)
            else:  # wall
                occupancy[y, x] = False
                wall_absorption[y, x] = absorption
    
    # Flip y-axis so origin is at bottom-left (matches FDTD convention)
    occupancy = np.flipud(occupancy)
    wall_absorption = np.flipud(wall_absorption)
    
    if source_pos:
        source_pos = (source_pos[0], height - 1 - source_pos[1])
    if mic_pos:
        mic_pos = (mic_pos[0], height - 1 - mic_pos[1])
    
    print(f"Parsed room: {width}x{height} pixels")
    print(f"  Air cells: {np.sum(occupancy)}")
    print(f"  Wall cells: {np.sum(~occupancy)}")
    print(f"  Source: {source_pos}")
    print(f"  Microphone: {mic_pos}")
    
    return RoomGeometry(
        occupancy=occupancy,
        wall_absorption=wall_absorption,
        source_pos=source_pos,
        mic_pos=mic_pos,
        width=width,
        height=height,
    )


# =============================================================================
# Wall Absorption Analysis
# =============================================================================

def get_edge_wall_absorption(
    geometry: RoomGeometry,
    rect_x: int, rect_y: int, 
    rect_width: int, rect_height: int,
    edge: str
) -> float:
    """Get the average wall absorption coefficient for an edge of a rectangle.
    
    Looks at the wall pixels just outside the rectangle edge.
    
    Returns:
        Average absorption coefficient (0 = reflective, 1 = absorbing)
    """
    wall_abs = geometry.wall_absorption
    occ = geometry.occupancy
    
    absorptions = []
    
    if edge == 'top':
        y = rect_y + rect_height  # Row above the rectangle
        if y < geometry.height:
            for x in range(rect_x, min(rect_x + rect_width, geometry.width)):
                if not occ[y, x]:  # It's a wall
                    absorptions.append(wall_abs[y, x])
    
    elif edge == 'bottom':
        y = rect_y - 1  # Row below the rectangle
        if y >= 0:
            for x in range(rect_x, min(rect_x + rect_width, geometry.width)):
                if not occ[y, x]:
                    absorptions.append(wall_abs[y, x])
    
    elif edge == 'left':
        x = rect_x - 1  # Column to the left
        if x >= 0:
            for y in range(rect_y, min(rect_y + rect_height, geometry.height)):
                if not occ[y, x]:
                    absorptions.append(wall_abs[y, x])
    
    elif edge == 'right':
        x = rect_x + rect_width  # Column to the right
        if x < geometry.width:
            for y in range(rect_y, min(rect_y + rect_height, geometry.height)):
                if not occ[y, x]:
                    absorptions.append(wall_abs[y, x])
    
    if absorptions:
        return float(np.mean(absorptions))
    return 0.0  # Default to fully reflective if no wall found


# =============================================================================
# Simulation Setup
# =============================================================================

def create_simulation(
    geometry: RoomGeometry,
    scale: float,
    f_max: float,
    courant: float,
    amplitude: float = 0.01,
) -> Tuple[Simulation, List[FDTDGrid], Optional[Microphone]]:
    """Create FDTD simulation from room geometry.
    
    Decomposes the room into rectangular regions connected by interfaces.
    Wall absorption is applied via coupling strength at boundaries.
    
    Args:
        geometry: Parsed room geometry
        scale: Meters per pixel
        f_max: Maximum frequency (Hz)
        courant: Courant number
        amplitude: Source amplitude
        
    Returns:
        Tuple of (simulation, list of grids, microphone)
    """
    sim = Simulation(f_max=f_max, courant=courant)
    params = sim.params
    
    room_width = geometry.width * scale
    room_height = geometry.height * scale
    
    print(f"\nSimulation setup:")
    print(f"  Room size: {room_width:.2f} x {room_height:.2f} m")
    print(f"  Grid spacing: {params.h:.4f} m")
    print(f"  Time step: {params.dt:.6f} s")
    print(f"  Sample rate: {1/params.dt:.0f} Hz")
    
    # Decompose room into rectangles
    rectangles = decompose_simple(geometry.occupancy, random_seed=42)
    rect_interfaces = extract_interfaces(rectangles)
    
    print(f"\nRoom decomposition: {len(rectangles)} regions, {len(rect_interfaces)} interfaces")
    
    # Create FDTD grid for each rectangle
    grids: List[FDTDGrid] = []
    for i, rect in enumerate(rectangles):
        rect_width = rect.width * scale
        rect_height = rect.height * scale
        
        grid = FDTDGrid(x_size=rect_width, y_size=rect_height, params=params)
        grid.rect = rect  # type: ignore
        grid.origin_x = rect.x * scale  # type: ignore
        grid.origin_y = rect.y * scale  # type: ignore
        
        grids.append(grid)
        sim.add_simulation_component(grid)
        
        print(f"  Region {i}: ({rect.x}, {rect.y}) {rect.width}x{rect.height} px -> {grid.xs}x{grid.ys} cells")
    
    # Track which edges have interfaces
    edge_has_interface = {i: {'top': set(), 'bottom': set(), 'left': set(), 'right': set()} 
                          for i in range(len(grids))}
    
    # Create interfaces between adjacent grids (full coupling for air-to-air)
    h = params.h
    for iface in rect_interfaces:
        grid_a = grids[iface.rect_a]
        grid_b = grids[iface.rect_b]
        rect_a = rectangles[iface.rect_a]
        rect_b = rectangles[iface.rect_b]
        
        if iface.edge_a in ('top', 'bottom'):
            start_px_a = iface.start
            end_px_a = start_px_a + iface.length
            start_cells_a = int(start_px_a * scale / h)
            end_cells_a = int(end_px_a * scale / h)
            
            abs_start = rect_a.x + iface.start
            start_px_b = abs_start - rect_b.x
            end_px_b = start_px_b + iface.length
            start_cells_b = int(start_px_b * scale / h)
            end_cells_b = int(end_px_b * scale / h)
            
            width_a = end_cells_a - start_cells_a
            width_b = end_cells_b - start_cells_b
            min_width = min(width_a, width_b)
            if min_width <= 0:
                continue
            end_cells_a = start_cells_a + min_width
            end_cells_b = start_cells_b + min_width
            
            region_a = (start_cells_a, end_cells_a)
            region_b = (start_cells_b, end_cells_b)
            
            # Mark these cells as having an interface
            for c in range(start_cells_a, end_cells_a):
                edge_has_interface[iface.rect_a][iface.edge_a].add(c)
            for c in range(start_cells_b, end_cells_b):
                edge_has_interface[iface.rect_b][iface.edge_b].add(c)
        else:
            start_px_a = iface.start
            end_px_a = start_px_a + iface.length
            start_cells_a = int(start_px_a * scale / h)
            end_cells_a = int(end_px_a * scale / h)
            
            abs_start = rect_a.y + iface.start
            start_px_b = abs_start - rect_b.y
            end_px_b = start_px_b + iface.length
            start_cells_b = int(start_px_b * scale / h)
            end_cells_b = int(end_px_b * scale / h)
            
            height_a = end_cells_a - start_cells_a
            height_b = end_cells_b - start_cells_b
            min_height = min(height_a, height_b)
            if min_height <= 0:
                continue
            end_cells_a = start_cells_a + min_height
            end_cells_b = start_cells_b + min_height
            
            region_a = (start_cells_a, end_cells_a)
            region_b = (start_cells_b, end_cells_b)
            
            for c in range(start_cells_a, end_cells_a):
                edge_has_interface[iface.rect_a][iface.edge_a].add(c)
            for c in range(start_cells_b, end_cells_b):
                edge_has_interface[iface.rect_b][iface.edge_b].add(c)
        
        try:
            sim.add_interface(iface.edge_a, grid_a, iface.edge_b, grid_b,
                            region_a=region_a, region_b=region_b)
        except ValueError as e:
            print(f"  Warning: Interface error: {e}")
    
    # For edges touching walls, add PMLs with coupling based on wall absorption.
    # Black walls (absorption=0) -> coupling=1 but PML absorbs nothing (essentially reflective)
    # Gray walls (absorption>0) -> coupling based on absorption, PML absorbs outgoing sound
    pml_thickness = 5
    pml_coeff = 800  # Base PML absorption coefficient
    
    for grid_idx, grid in enumerate(grids):
        rect = rectangles[grid_idx]
        
        for edge in ['top', 'bottom', 'left', 'right']:
            # Check if this edge needs a wall PML (not fully covered by air interfaces)
            edge_dim = grid.xs if edge in ('top', 'bottom') else grid.ys
            covered_cells = edge_has_interface[grid_idx][edge]
            
            # If entire edge is covered by air interfaces, skip
            if len(covered_cells) == edge_dim:
                continue
            
            # Get wall absorption for this edge
            wall_abs = get_edge_wall_absorption(geometry, rect.x, rect.y, rect.width, rect.height, edge)
            
            # Create PML for this edge
            # Direction is where sound comes FROM (into the PML)
            if edge == 'top':
                direction = 'down'
                width = grid.xs
            elif edge == 'bottom':
                direction = 'up'
                width = grid.xs
            elif edge == 'left':
                direction = 'right'
                width = grid.ys
            else:  # right
                direction = 'left'
                width = grid.ys
            
            
            pml = PML(
                thickness=pml_thickness,
                width=width,
                direction=direction,
                pml_coeff=pml_coeff,
                params=params
            )
            sim.add_simulation_component(pml)
            
            # Coupling strength: low absorption = low coupling (more reflection)
            # high absorption = high coupling (sound enters PML and gets absorbed)
            coupling = 0.1 + 0.9 * wall_abs  # Range: 0.1 to 1.0
            
            try:
                # Create interface between grid edge and PML
                # PML edge is opposite to grid edge
                pml_edge = {'top': 'bottom', 'bottom': 'top', 'left': 'right', 'right': 'left'}[edge]
                iface = Interface(
                    edge, grid, pml_edge, pml, params,
                    coupling_a2b=coupling, coupling_b2a=coupling
                )
                sim.interfaces.append(iface)
                # Register with grid and PML
                grid.add_interface(edge, iface)
                pml.interface = iface
            except Exception as e:
                print(f"  Warning: Could not add wall PML for region {grid_idx} {edge}: {e}")
    
    # Helper to find grid containing a point
    def find_grid_for_point(px: int, py: int) -> Optional[Tuple[FDTDGrid, float, float]]:
        for grid in grids:
            rect = grid.rect  # type: ignore
            if rect.contains(px, py):
                local_x = (px - rect.x + 0.5) * scale
                local_y = (py - rect.y + 0.5) * scale
                return grid, local_x, local_y
        return None
    
    # Add source
    if geometry.source_pos:
        px, py = geometry.source_pos
        result = find_grid_for_point(px, py)
        if result:
            grid, local_x, local_y = result
            source = RickerWaveletSource(local_x, local_y, f_peak=f_max / 2.0, amplitude=amplitude)
            grid.add_source(source)
            global_x = grid.origin_x + local_x  # type: ignore
            global_y = grid.origin_y + local_y  # type: ignore
            print(f"\n  Source at ({global_x:.2f}, {global_y:.2f}) m")
        else:
            print(f"\n  Warning: Source at {geometry.source_pos} not in air region")
    
    # Add microphone
    microphone: Optional[Microphone] = None
    if geometry.mic_pos:
        px, py = geometry.mic_pos
        result = find_grid_for_point(px, py)
        if result:
            grid, local_x, local_y = result
            microphone = Microphone(local_x, local_y, name="mic")
            grid.add_microphone(microphone)
            global_x = grid.origin_x + local_x  # type: ignore
            global_y = grid.origin_y + local_y  # type: ignore
            print(f"  Microphone at ({global_x:.2f}, {global_y:.2f}) m")
        else:
            print(f"  Warning: Microphone at {geometry.mic_pos} not in air region")
    
    return sim, grids, microphone


# =============================================================================
# Simulation Runner
# =============================================================================

def run_simulation(
    sim: Simulation,
    grids: List[FDTDGrid],
    duration: float,
) -> None:
    """Run the simulation."""
    params = sim.params
    total_steps = int(duration / params.dt)
    
    print(f"\nRunning simulation:")
    print(f"  Duration: {duration:.2f} s")
    print(f"  Total steps: {total_steps}")
    
    for step in range(total_steps):
        sim.step()
        
        if step % 1000 == 0:
            t = step * params.dt
            max_p = max(np.max(np.abs(grid.p_1)) for grid in grids)
            print(f"  Step {step}/{total_steps} (t = {t:.3f} s), max |p| = {max_p:.4f}")
    
    print(f"  Simulation complete!")


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
                       help='Output WAV file path')
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
    
    args = parser.parse_args()
    
    # Parse room geometry
    geometry = parse_room_png(args.room_png)
    
    if not geometry.source_pos:
        print("\nError: No source found in PNG!")
        print("Add a red (255,0,0) pixel to mark the source position.")
        return
    
    if not geometry.mic_pos:
        print("\nError: No microphone found in PNG!")
        print("Add a cyan (0,255,255) pixel to mark the microphone position.")
        return
    
    # Create and run simulation
    sim, grids, microphone = create_simulation(
        geometry=geometry,
        scale=args.scale,
        f_max=args.f_max,
        courant=args.courant,
        amplitude=args.amplitude,
    )
    
    if microphone is None:
        print("\nError: Microphone could not be placed.")
        return
    
    run_simulation(sim=sim, grids=grids, duration=args.duration)
    
    # Save impulse response
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    microphone.save_wav(args.output, target_sr=args.output_sr, normalize=True)
    
    print(f"\nDone! Impulse response saved to {args.output}")


if __name__ == '__main__':
    main()
