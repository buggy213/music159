"""
Unit tests for Adaptive Room Decomposition (ARD) module.
"""

import numpy as np
import pytest
from ard import (
    Rectangle, RectangleInterface,
    is_power_of_two, next_power_of_two, prev_power_of_two,
    find_largest_pow2_rect, decompose, extract_interfaces
)


class TestPowerOfTwoHelpers:
    """Tests for power-of-two helper functions."""
    
    def test_is_power_of_two_true(self):
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            assert is_power_of_two(n), f"{n} should be power of two"
    
    def test_is_power_of_two_false(self):
        for n in [0, 3, 5, 6, 7, 9, 10, 15, 17, 100]:
            assert not is_power_of_two(n), f"{n} should not be power of two"
    
    def test_next_power_of_two(self):
        assert next_power_of_two(0) == 1
        assert next_power_of_two(1) == 1
        assert next_power_of_two(2) == 2
        assert next_power_of_two(3) == 4
        assert next_power_of_two(4) == 4
        assert next_power_of_two(5) == 8
        assert next_power_of_two(7) == 8
        assert next_power_of_two(8) == 8
        assert next_power_of_two(9) == 16
        assert next_power_of_two(100) == 128
    
    def test_prev_power_of_two(self):
        assert prev_power_of_two(0) == 0
        assert prev_power_of_two(1) == 1
        assert prev_power_of_two(2) == 2
        assert prev_power_of_two(3) == 2
        assert prev_power_of_two(4) == 4
        assert prev_power_of_two(5) == 4
        assert prev_power_of_two(7) == 4
        assert prev_power_of_two(8) == 8
        assert prev_power_of_two(9) == 8
        assert prev_power_of_two(100) == 64


class TestRectangle:
    """Tests for Rectangle dataclass."""
    
    def test_basic_properties(self):
        rect = Rectangle(x=2, y=3, width=4, height=8)
        assert rect.x == 2
        assert rect.y == 3
        assert rect.width == 4
        assert rect.height == 8
        assert rect.x2 == 6
        assert rect.y2 == 11
        assert rect.area == 32
    
    def test_contains(self):
        rect = Rectangle(x=2, y=3, width=4, height=4)
        # Inside
        assert rect.contains(2, 3)
        assert rect.contains(5, 6)
        assert rect.contains(3, 4)
        # Outside
        assert not rect.contains(1, 3)  # Left of
        assert not rect.contains(6, 3)  # Right edge (exclusive)
        assert not rect.contains(2, 7)  # Top edge (exclusive)
        assert not rect.contains(2, 2)  # Below
    
    def test_overlaps(self):
        rect1 = Rectangle(x=0, y=0, width=4, height=4)
        rect2 = Rectangle(x=2, y=2, width=4, height=4)  # Overlapping
        rect3 = Rectangle(x=4, y=0, width=4, height=4)  # Adjacent (no overlap)
        rect4 = Rectangle(x=5, y=0, width=4, height=4)  # Separated
        
        assert rect1.overlaps(rect2)
        assert rect2.overlaps(rect1)
        assert not rect1.overlaps(rect3)  # Adjacent edges don't overlap
        assert not rect1.overlaps(rect4)


class TestDecomposition:
    """Tests for the main decomposition algorithm."""
    
    def test_empty_grid(self):
        """Empty grid should return no rectangles."""
        grid = np.zeros((8, 8), dtype=bool)
        rects = decompose(grid)
        assert len(rects) == 0
    
    def test_full_power_of_two_grid_ordered(self):
        """Full power-of-two grid with ordered strategy should yield single rectangle."""
        grid = np.ones((8, 8), dtype=bool)
        rects = decompose(grid, strategy='ordered')
        assert len(rects) == 1
        assert rects[0].width == 8
        assert rects[0].height == 8
    
    def test_full_power_of_two_grid_random(self):
        """Full power-of-two grid with random strategy should still cover everything."""
        grid = np.ones((8, 8), dtype=bool)
        rects = decompose(grid, random_seed=42)
        
        # Verify full coverage
        covered = np.zeros_like(grid, dtype=bool)
        for rect in rects:
            covered[rect.y:rect.y2, rect.x:rect.x2] = True
        assert np.all(covered == grid)
        
        # All rectangles should be power-of-two
        for rect in rects:
            assert is_power_of_two(rect.width)
            assert is_power_of_two(rect.height)
    
    def test_non_power_of_two_grid(self):
        """Non-power-of-two grid should still be fully covered."""
        grid = np.ones((10, 12), dtype=bool)
        rects = decompose(grid, random_seed=42)
        
        # Verify full coverage
        covered = np.zeros_like(grid, dtype=bool)
        for rect in rects:
            covered[rect.y:rect.y2, rect.x:rect.x2] = True
        
        assert np.all(covered == grid)
        
        # Verify all rectangles have power-of-two dimensions
        for rect in rects:
            assert is_power_of_two(rect.width), f"width {rect.width} not power of two"
            assert is_power_of_two(rect.height), f"height {rect.height} not power of two"
    
    def test_l_shaped_room(self):
        """L-shaped room should decompose correctly."""
        grid = np.zeros((16, 16), dtype=bool)
        grid[0:8, 0:8] = True   # Bottom-left square
        grid[0:4, 8:16] = True  # Bottom-right extension
        
        rects = decompose(grid, random_seed=42)
        
        # Verify full coverage
        covered = np.zeros_like(grid, dtype=bool)
        for rect in rects:
            covered[rect.y:rect.y2, rect.x:rect.x2] = True
        
        assert np.all(covered == grid)
        
        # Verify no overlaps
        for i, r1 in enumerate(rects):
            for j, r2 in enumerate(rects):
                if i < j:
                    assert not r1.overlaps(r2), f"rectangles {i} and {j} overlap"
    
    def test_complex_shape(self):
        """Complex shape with multiple concavities."""
        grid = np.zeros((32, 32), dtype=bool)
        # Create a plus-shaped room
        grid[12:20, 4:28] = True   # Horizontal bar
        grid[4:28, 12:20] = True   # Vertical bar
        
        rects = decompose(grid, random_seed=42)
        
        # Verify full coverage
        covered = np.zeros_like(grid, dtype=bool)
        for rect in rects:
            covered[rect.y:rect.y2, rect.x:rect.x2] = True
        
        assert np.sum(covered & grid) == np.sum(grid)
        
        # Verify power-of-two dimensions
        for rect in rects:
            assert is_power_of_two(rect.width)
            assert is_power_of_two(rect.height)
    
    def test_single_cell(self):
        """Single empty cell should yield 1x1 rectangle."""
        grid = np.zeros((8, 8), dtype=bool)
        grid[3, 4] = True
        
        rects = decompose(grid, random_seed=42)
        assert len(rects) == 1
        assert rects[0].width == 1
        assert rects[0].height == 1
        assert rects[0].x == 4
        assert rects[0].y == 3
    
    def test_scattered_cells(self):
        """Scattered single cells should each get 1x1 rectangle."""
        grid = np.zeros((8, 8), dtype=bool)
        grid[1, 1] = True
        grid[1, 6] = True
        grid[6, 1] = True
        grid[6, 6] = True
        
        rects = decompose(grid, random_seed=42)
        assert len(rects) == 4
        for rect in rects:
            assert rect.width == 1
            assert rect.height == 1
    
    def test_random_reproducible(self):
        """Same seed should produce same decomposition."""
        grid = np.ones((16, 16), dtype=bool)
        grid[0:4, 0:4] = False  # Cut out corner
        
        rects1 = decompose(grid, random_seed=123)
        rects2 = decompose(grid, random_seed=123)
        
        assert len(rects1) == len(rects2)
        for r1, r2 in zip(rects1, rects2):
            assert r1.x == r2.x
            assert r1.y == r2.y
            assert r1.width == r2.width
            assert r1.height == r2.height
    
    def test_ordered_vs_random_both_cover(self):
        """Both strategies should cover the same cells."""
        grid = np.zeros((16, 16), dtype=bool)
        grid[2:14, 2:8] = True
        grid[2:8, 8:14] = True
        
        rects_ordered = decompose(grid, strategy='ordered')
        rects_random = decompose(grid, random_seed=42, strategy='random')
        
        covered_ordered = np.zeros_like(grid, dtype=bool)
        covered_random = np.zeros_like(grid, dtype=bool)
        
        for rect in rects_ordered:
            covered_ordered[rect.y:rect.y2, rect.x:rect.x2] = True
        for rect in rects_random:
            covered_random[rect.y:rect.y2, rect.x:rect.x2] = True
        
        assert np.all(covered_ordered == grid)
        assert np.all(covered_random == grid)


class TestInterfaceExtraction:
    """Tests for interface extraction between rectangles."""
    
    def test_horizontal_adjacent(self):
        """Two horizontally adjacent rectangles."""
        rects = [
            Rectangle(x=0, y=0, width=4, height=4),
            Rectangle(x=4, y=0, width=4, height=4),
        ]
        
        interfaces = extract_interfaces(rects)
        assert len(interfaces) == 1
        
        iface = interfaces[0]
        assert iface.rect_a == 0
        assert iface.rect_b == 1
        assert iface.edge_a == 'right'
        assert iface.edge_b == 'left'
        assert iface.length == 4
    
    def test_vertical_adjacent(self):
        """Two vertically adjacent rectangles."""
        rects = [
            Rectangle(x=0, y=0, width=4, height=4),
            Rectangle(x=0, y=4, width=4, height=4),
        ]
        
        interfaces = extract_interfaces(rects)
        assert len(interfaces) == 1
        
        iface = interfaces[0]
        assert iface.rect_a == 0
        assert iface.rect_b == 1
        assert iface.edge_a == 'top'
        assert iface.edge_b == 'bottom'
        assert iface.length == 4
    
    def test_partial_overlap_interface(self):
        """Rectangles with partial edge overlap."""
        rects = [
            Rectangle(x=0, y=0, width=4, height=8),
            Rectangle(x=4, y=2, width=4, height=4),
        ]
        
        interfaces = extract_interfaces(rects)
        assert len(interfaces) == 1
        
        iface = interfaces[0]
        assert iface.length == 4  # Only 4 cells overlap
    
    def test_no_interface_separated(self):
        """Separated rectangles should have no interface."""
        rects = [
            Rectangle(x=0, y=0, width=4, height=4),
            Rectangle(x=8, y=0, width=4, height=4),  # Gap of 4 cells
        ]
        
        interfaces = extract_interfaces(rects)
        assert len(interfaces) == 0
    
    def test_corner_touch_no_interface(self):
        """Rectangles touching only at corners should have no interface."""
        rects = [
            Rectangle(x=0, y=0, width=4, height=4),
            Rectangle(x=4, y=4, width=4, height=4),  # Diagonal
        ]
        
        interfaces = extract_interfaces(rects)
        assert len(interfaces) == 0
    
    def test_three_rectangles(self):
        """Three rectangles in a row."""
        rects = [
            Rectangle(x=0, y=0, width=4, height=4),
            Rectangle(x=4, y=0, width=4, height=4),
            Rectangle(x=8, y=0, width=4, height=4),
        ]
        
        interfaces = extract_interfaces(rects)
        assert len(interfaces) == 2


class TestFindLargestPow2Rect:
    """Tests for the rectangle growing function."""
    
    def test_grow_in_empty_space(self):
        """Should grow to fill available space with power-of-two sizes."""
        grid = np.ones((8, 8), dtype=bool)
        covered = np.zeros_like(grid, dtype=bool)
        
        rect = find_largest_pow2_rect(grid, covered, 0, 0)
        
        assert is_power_of_two(rect.width)
        assert is_power_of_two(rect.height)
        assert rect.width == 8
        assert rect.height == 8
    
    def test_grow_limited_by_wall(self):
        """Growth should stop at walls."""
        grid = np.ones((8, 8), dtype=bool)
        grid[:, 4:] = False  # Wall on right half
        covered = np.zeros_like(grid, dtype=bool)
        
        rect = find_largest_pow2_rect(grid, covered, 0, 0)
        
        assert rect.width == 4  # Limited by wall
        assert rect.height == 8
    
    def test_grow_limited_by_covered(self):
        """Growth should stop at already-covered cells."""
        grid = np.ones((8, 8), dtype=bool)
        covered = np.zeros_like(grid, dtype=bool)
        covered[:, 4:] = True  # Right half already covered
        
        rect = find_largest_pow2_rect(grid, covered, 0, 0)
        
        assert rect.width == 4  # Limited by covered region
        assert rect.height == 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
