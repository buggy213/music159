"""
Adaptive Room Decomposition (ARD) Algorithm

Decomposes a binary occupancy grid into non-overlapping, axis-aligned rectangles
with power-of-two dimensions for efficient spectral acoustic simulation.

Input: 2D numpy boolean array (True = empty/air, False = wall/occupied)
Output: List of Rectangle objects that cover all empty cells

Designed for 2D initially, with 3D extension in mind.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Rectangle:
    """A rectangular region in the decomposition.
    
    Attributes:
        x: Left column (grid coords)
        y: Bottom row (grid coords)
        width: Width in cells (power of two)
        height: Height in cells (power of two)
    """
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        """Right edge (exclusive)."""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """Top edge (exclusive)."""
        return self.y + self.height
    
    @property
    def area(self) -> int:
        """Area in cells."""
        return self.width * self.height
    
    def contains(self, px: int, py: int) -> bool:
        """Check if point is inside this rectangle."""
        return self.x <= px < self.x2 and self.y <= py < self.y2
    
    def overlaps(self, other: 'Rectangle') -> bool:
        """Check if this rectangle overlaps with another."""
        return not (self.x2 <= other.x or other.x2 <= self.x or
                    self.y2 <= other.y or other.y2 <= self.y)


@dataclass
class RectangleInterface:
    """Interface between two adjacent rectangles.
    
    Attributes:
        rect_a: Index of first rectangle
        rect_b: Index of second rectangle
        edge_a: Edge of rect_a ('top', 'bottom', 'left', 'right')
        edge_b: Edge of rect_b (opposite of edge_a)
        start: Start position along the edge (in rect_a's local coords)
        length: Length of the shared interface
    """
    rect_a: int
    rect_b: int
    edge_a: str
    edge_b: str
    start: int
    length: int


# =============================================================================
# Power-of-two helpers
# =============================================================================

def is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 0:
        return 1
    if is_power_of_two(n):
        return n
    # Find next power of two
    p = 1
    while p < n:
        p *= 2
    return p


def prev_power_of_two(n: int) -> int:
    """Return the largest power of two <= n."""
    if n <= 0:
        return 0
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


# =============================================================================
# Rectangle growing
# =============================================================================

def _can_expand(grid: np.ndarray, covered: np.ndarray,
                x: int, y: int, width: int, height: int,
                direction: str, amount: int) -> bool:
    """Check if we can expand the rectangle in a given direction.
    
    Args:
        grid: Binary occupancy grid (True = empty)
        covered: Boolean array tracking already-covered cells
        x, y: Current rectangle origin (bottom-left)
        width, height: Current rectangle dimensions
        direction: 'left', 'right', 'up', or 'down'
        amount: Number of cells to expand by
    
    Returns:
        True if expansion is valid
    """
    if direction == 'right':
        new_x = x + width
        new_x2 = new_x + amount
        if new_x2 > grid.shape[1]:
            return False
        region = grid[y:y+height, new_x:new_x2]
        covered_region = covered[y:y+height, new_x:new_x2]
    
    elif direction == 'left':
        new_x = x - amount
        if new_x < 0:
            return False
        region = grid[y:y+height, new_x:x]
        covered_region = covered[y:y+height, new_x:x]
    
    elif direction == 'up':
        new_y = y + height
        new_y2 = new_y + amount
        if new_y2 > grid.shape[0]:
            return False
        region = grid[new_y:new_y2, x:x+width]
        covered_region = covered[new_y:new_y2, x:x+width]
    
    elif direction == 'down':
        new_y = y - amount
        if new_y < 0:
            return False
        region = grid[new_y:y, x:x+width]
        covered_region = covered[new_y:y, x:x+width]
    
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    return np.all(region) and not np.any(covered_region)


def _find_max_extent(grid: np.ndarray, covered: np.ndarray,
                     seed_x: int, seed_y: int) -> Tuple[int, int, int, int]:
    """Find maximum extent from seed in all 4 directions.
    
    Returns:
        (left, right, down, up) - number of cells we can extend in each direction
    """
    h, w = grid.shape
    
    # Find extent right
    right = 0
    for x in range(seed_x + 1, w):
        if grid[seed_y, x] and not covered[seed_y, x]:
            right += 1
        else:
            break
    
    # Find extent left
    left = 0
    for x in range(seed_x - 1, -1, -1):
        if grid[seed_y, x] and not covered[seed_y, x]:
            left += 1
        else:
            break
    
    # Find extent up
    up = 0
    for y in range(seed_y + 1, h):
        if grid[y, seed_x] and not covered[y, seed_x]:
            up += 1
        else:
            break
    
    # Find extent down
    down = 0
    for y in range(seed_y - 1, -1, -1):
        if grid[y, seed_x] and not covered[y, seed_x]:
            down += 1
        else:
            break
    
    return left, right, down, up


def find_largest_pow2_rect(grid: np.ndarray, covered: np.ndarray, 
                           seed_x: int, seed_y: int,
                           expand_all_directions: bool = True) -> Rectangle:
    """Grow the largest power-of-two rectangle from a seed cell.
    
    Args:
        grid: Binary occupancy grid (True = empty)
        covered: Boolean array tracking already-covered cells
        seed_x: Starting x coordinate
        seed_y: Starting y coordinate
        expand_all_directions: If True, expand in all 4 directions (for random seeds).
                               If False, only expand right/up (for ordered scanning).
    
    Returns:
        Rectangle with power-of-two dimensions
    """
    if expand_all_directions:
        # Find maximum extent in all directions
        left_ext, right_ext, down_ext, up_ext = _find_max_extent(grid, covered, seed_x, seed_y)
        
        # Total possible width and height
        total_width = left_ext + 1 + right_ext  # +1 for seed cell
        total_height = down_ext + 1 + up_ext
        
        # Find largest power-of-two that fits
        target_width = prev_power_of_two(total_width)
        target_height = prev_power_of_two(total_height)
        
        # Center the rectangle as much as possible within the extent
        # For width: we have left_ext cells to the left, right_ext to the right
        # We need (target_width - 1) additional cells beyond the seed
        if target_width == 1:
            x = seed_x
        else:
            # Distribute the rectangle around the seed
            # Try to take half from left, half from right
            needed = target_width - 1
            take_left = min(left_ext, needed // 2)
            take_right = min(right_ext, needed - take_left)
            # If we couldn't take enough from right, take more from left
            if take_right < needed - take_left:
                take_left = min(left_ext, needed - take_right)
            x = seed_x - take_left
        
        if target_height == 1:
            y = seed_y
        else:
            needed = target_height - 1
            take_down = min(down_ext, needed // 2)
            take_up = min(up_ext, needed - take_down)
            if take_up < needed - take_down:
                take_down = min(down_ext, needed - take_up)
            y = seed_y - take_down
        
        # Now verify and expand using the standard method
        width, height = target_width, target_height
        
        # Verify the rectangle is valid (all cells empty and uncovered)
        if not np.all(grid[y:y+height, x:x+width]) or np.any(covered[y:y+height, x:x+width]):
            # Fallback to conservative 1x1 expansion
            x, y = seed_x, seed_y
            width, height = 1, 1
        
        # Try to expand further using doubling
        expanded = True
        while expanded:
            expanded = False
            
            # Try doubling width
            if _can_expand(grid, covered, x, y, width, height, 'right', width):
                width *= 2
                expanded = True
            elif _can_expand(grid, covered, x, y, width, height, 'left', width):
                x -= width
                width *= 2
                expanded = True
            
            # Try doubling height
            if _can_expand(grid, covered, x, y, width, height, 'up', height):
                height *= 2
                expanded = True
            elif _can_expand(grid, covered, x, y, width, height, 'down', height):
                y -= height
                height *= 2
                expanded = True
        
        return Rectangle(x=x, y=y, width=width, height=height)
    
    else:
        # Original ordered strategy: only expand right/up
        x, y = seed_x, seed_y
        width, height = 1, 1
        
        expanded = True
        while expanded:
            expanded = False
            
            if _can_expand(grid, covered, x, y, width, height, 'right', width):
                width *= 2
                expanded = True
            
            if _can_expand(grid, covered, x, y, width, height, 'up', height):
                height *= 2
                expanded = True
        
        return Rectangle(x=x, y=y, width=width, height=height)


# =============================================================================
# Main decomposition
# =============================================================================

def decompose(grid: np.ndarray, 
              random_seed: Optional[int] = None,
              strategy: str = 'random') -> List[Rectangle]:
    """Decompose a binary occupancy grid into power-of-two rectangles.
    
    Args:
        grid: 2D boolean numpy array (True = empty/air, False = wall/occupied)
        random_seed: Seed for reproducible random decomposition (None for non-deterministic)
        strategy: Seed selection strategy:
            - 'random': Random seed selection, expand in all 4 directions (often better)
            - 'ordered': Top-left to bottom-right scanning, expand right/up only (deterministic)
    
    Returns:
        List of non-overlapping Rectangle objects covering all empty cells
    """
    if grid.ndim != 2:
        raise ValueError(f"Expected 2D grid, got {grid.ndim}D")
    
    if strategy not in ('random', 'ordered'):
        raise ValueError(f"Unknown strategy: {strategy}. Use 'random' or 'ordered'.")
    
    grid = grid.astype(bool)
    covered = np.zeros_like(grid, dtype=bool)
    rectangles: List[Rectangle] = []
    
    # Find all empty cells
    empty_cells = np.argwhere(grid)
    
    if len(empty_cells) == 0:
        return rectangles
    
    # Set up RNG if using random strategy
    if strategy == 'random':
        rng = np.random.default_rng(random_seed)
        expand_all_directions = True
    else:
        rng = None
        expand_all_directions = False
        # Sort by y then x for deterministic ordering
        empty_cells = empty_cells[np.lexsort((empty_cells[:, 1], empty_cells[:, 0]))]
    
    # Track uncovered cells as a set for efficient random sampling
    uncovered_set = set(map(tuple, empty_cells))
    
    while uncovered_set:
        if strategy == 'random':
            # Pick a random uncovered cell
            cell = rng.choice(list(uncovered_set))
        else:
            # Pick the first uncovered cell in order
            cell = None
            for c in empty_cells:
                if tuple(c) in uncovered_set:
                    cell = c
                    break
            if cell is None:
                break
        
        y, x = cell[0], cell[1]
        
        # Grow rectangle from this seed
        rect = find_largest_pow2_rect(grid, covered, x, y, expand_all_directions)
        rectangles.append(rect)
        
        # Mark cells as covered and remove from uncovered set
        for ry in range(rect.y, rect.y2):
            for rx in range(rect.x, rect.x2):
                covered[ry, rx] = True
                uncovered_set.discard((ry, rx))
    
    return rectangles


# =============================================================================
# Interface extraction
# =============================================================================

def extract_interfaces(rectangles: List[Rectangle]) -> List[RectangleInterface]:
    """Find shared edges between adjacent rectangles.
    
    Args:
        rectangles: List of Rectangle objects from decomposition
    
    Returns:
        List of RectangleInterface objects describing shared edges
    """
    interfaces: List[RectangleInterface] = []
    
    for i, rect_a in enumerate(rectangles):
        for j, rect_b in enumerate(rectangles):
            if i >= j:
                continue  # Avoid duplicates
            
            # Check for horizontal adjacency (rect_a's right edge touches rect_b's left edge)
            if rect_a.x2 == rect_b.x:
                # Find vertical overlap
                y_start = max(rect_a.y, rect_b.y)
                y_end = min(rect_a.y2, rect_b.y2)
                if y_start < y_end:
                    interfaces.append(RectangleInterface(
                        rect_a=i,
                        rect_b=j,
                        edge_a='right',
                        edge_b='left',
                        start=y_start - rect_a.y,
                        length=y_end - y_start
                    ))
            
            # Check if rect_b's right edge touches rect_a's left edge
            if rect_b.x2 == rect_a.x:
                y_start = max(rect_a.y, rect_b.y)
                y_end = min(rect_a.y2, rect_b.y2)
                if y_start < y_end:
                    interfaces.append(RectangleInterface(
                        rect_a=i,
                        rect_b=j,
                        edge_a='left',
                        edge_b='right',
                        start=y_start - rect_a.y,
                        length=y_end - y_start
                    ))
            
            # Check for vertical adjacency (rect_a's top edge touches rect_b's bottom edge)
            if rect_a.y2 == rect_b.y:
                x_start = max(rect_a.x, rect_b.x)
                x_end = min(rect_a.x2, rect_b.x2)
                if x_start < x_end:
                    interfaces.append(RectangleInterface(
                        rect_a=i,
                        rect_b=j,
                        edge_a='top',
                        edge_b='bottom',
                        start=x_start - rect_a.x,
                        length=x_end - x_start
                    ))
            
            # Check if rect_b's top edge touches rect_a's bottom edge
            if rect_b.y2 == rect_a.y:
                x_start = max(rect_a.x, rect_b.x)
                x_end = min(rect_a.x2, rect_b.x2)
                if x_start < x_end:
                    interfaces.append(RectangleInterface(
                        rect_a=i,
                        rect_b=j,
                        edge_a='bottom',
                        edge_b='top',
                        start=x_start - rect_a.x,
                        length=x_end - x_start
                    ))
    
    return interfaces


# =============================================================================
# Visualization
# =============================================================================

def visualize_decomposition(grid: np.ndarray, rectangles: List[Rectangle],
                            interfaces: Optional[List[RectangleInterface]] = None,
                            ax=None, show: bool = True):
    """Visualize the decomposition result.
    
    Args:
        grid: Original occupancy grid
        rectangles: List of rectangles from decomposition
        interfaces: Optional list of interfaces to highlight
        ax: Matplotlib axes (creates new figure if None)
        show: Whether to call plt.show()
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import hsv_to_rgb
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid background (walls in dark gray, empty in white)
    ax.imshow(~grid, cmap='gray', origin='lower', extent=[0, grid.shape[1], 0, grid.shape[0]])
    
    # Draw rectangles with different colors
    for i, rect in enumerate(rectangles):
        # Generate distinct color using golden ratio
        hue = (i * 0.618033988749895) % 1.0
        color = hsv_to_rgb([hue, 0.6, 0.9])
        
        patch = patches.Rectangle(
            (rect.x, rect.y), rect.width, rect.height,
            linewidth=2, edgecolor='black', facecolor=color, alpha=0.5
        )
        ax.add_patch(patch)
        
        # Label with index and dimensions
        cx = rect.x + rect.width / 2
        cy = rect.y + rect.height / 2
        ax.text(cx, cy, f'{i}\n{rect.width}x{rect.height}', 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw interfaces as thick lines
    if interfaces:
        for iface in interfaces:
            rect_a = rectangles[iface.rect_a]
            rect_b = rectangles[iface.rect_b]
            
            if iface.edge_a == 'right':
                x = rect_a.x2
                y1 = rect_a.y + iface.start
                y2 = y1 + iface.length
                ax.plot([x, x], [y1, y2], 'r-', linewidth=3, alpha=0.7)
            elif iface.edge_a == 'left':
                x = rect_a.x
                y1 = rect_a.y + iface.start
                y2 = y1 + iface.length
                ax.plot([x, x], [y1, y2], 'r-', linewidth=3, alpha=0.7)
            elif iface.edge_a == 'top':
                y = rect_a.y2
                x1 = rect_a.x + iface.start
                x2 = x1 + iface.length
                ax.plot([x1, x2], [y, y], 'r-', linewidth=3, alpha=0.7)
            elif iface.edge_a == 'bottom':
                y = rect_a.y
                x1 = rect_a.x + iface.start
                x2 = x1 + iface.length
                ax.plot([x1, x2], [y, y], 'r-', linewidth=3, alpha=0.7)
    
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Room Decomposition: {len(rectangles)} rectangles')
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.show()
    
    return ax


# =============================================================================
# Demo / Test
# =============================================================================

def _demo_decompose(grid: np.ndarray, name: str, strategy: str, random_seed: Optional[int] = None):
    """Run decomposition and print results."""
    print(f"\n{'='*60}")
    print(f"{name} - Strategy: {strategy}" + (f" (seed={random_seed})" if random_seed else ""))
    print('='*60)
    
    rectangles = decompose(grid, random_seed=random_seed, strategy=strategy)
    
    print(f"Decomposition: {len(rectangles)} rectangles")
    for i, rect in enumerate(rectangles):
        print(f"  {i}: x={rect.x}, y={rect.y}, {rect.width}x{rect.height} (area={rect.area})")
    
    # Verify coverage
    covered = np.zeros_like(grid, dtype=bool)
    for rect in rectangles:
        covered[rect.y:rect.y2, rect.x:rect.x2] = True
    
    empty_count = np.sum(grid)
    covered_count = np.sum(covered & grid)
    print(f"Coverage: {covered_count}/{empty_count} empty cells covered")
    
    # Check for overlaps
    for i, r1 in enumerate(rectangles):
        for j, r2 in enumerate(rectangles):
            if i < j and r1.overlaps(r2):
                print(f"WARNING: rectangles {i} and {j} overlap!")
    
    # Extract interfaces
    interfaces = extract_interfaces(rectangles)
    print(f"Interfaces: {len(interfaces)}")
    
    return rectangles, interfaces


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create an L-shaped room
    grid = np.zeros((16, 16), dtype=bool)
    grid[2:14, 2:8] = True   # Left vertical section
    grid[2:8, 8:14] = True   # Bottom horizontal section
    
    print("Input grid (1=empty, 0=wall):")
    print(grid.astype(int))
    
    # Compare strategies
    rects_ordered, ifaces_ordered = _demo_decompose(grid, "L-shaped room", 'ordered')
    rects_random, ifaces_random = _demo_decompose(grid, "L-shaped room", 'random', random_seed=42)
    
    # Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    visualize_decomposition(grid, rects_ordered, ifaces_ordered, ax=axes[0], show=False)
    axes[0].set_title(f'Ordered Strategy: {len(rects_ordered)} rectangles')
    
    visualize_decomposition(grid, rects_random, ifaces_random, ax=axes[1], show=False)
    axes[1].set_title(f'Random Strategy (seed=42): {len(rects_random)} rectangles')
    
    plt.tight_layout()
    plt.savefig('output/ard_comparison.png', dpi=150)
    print("\nSaved comparison to output/ard_comparison.png")
    plt.show()
