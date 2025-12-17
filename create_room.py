#!/usr/bin/env python3
"""
Create example room PNGs for the acoustic simulator.

Usage:
    python create_room.py simple --output rooms/simple_room.png
    python create_room.py l_shaped --output rooms/l_shaped.png
    python create_room.py concert_hall --output rooms/concert_hall.png
"""

import argparse
import os
from PIL import Image, ImageDraw


# Color definitions (matching app.py)
WALL = (0, 0, 0)              # Fully reflective
AIR = (255, 255, 255)         # Empty space
SOURCE = (255, 0, 0)          # Red = source
MIC = (0, 255, 255)           # Cyan = microphone
ABSORBING_WALL = (128, 128, 128)  # Gray = 50% absorbing wall


def create_simple_room(width: int = 100, height: int = 80) -> Image.Image:
    """Create a simple rectangular room with one source and one microphone."""
    img = Image.new('RGB', (width, height), WALL)
    draw = ImageDraw.Draw(img)
    
    # Inner room (air)
    margin = 5
    draw.rectangle([margin, margin, width - margin - 1, height - margin - 1], fill=AIR)
    
    # Source in upper-left area
    img.putpixel((margin + 10, height - margin - 15), SOURCE)
    
    # Microphone in lower-right area
    img.putpixel((width - margin - 15, margin + 10), MIC)
    
    return img


def create_l_shaped_room(width: int = 120, height: int = 100) -> Image.Image:
    """Create an L-shaped room with source and microphone."""
    img = Image.new('RGB', (width, height), WALL)
    draw = ImageDraw.Draw(img)
    
    margin = 5
    
    # L-shape: vertical part (left side)
    draw.rectangle([margin, margin, width // 2, height - margin - 1], fill=AIR)
    
    # L-shape: horizontal part (bottom)
    draw.rectangle([margin, margin, width - margin - 1, height // 2], fill=AIR)
    
    # Source in the vertical part
    img.putpixel((margin + 15, height - margin - 20), SOURCE)
    
    # Microphone in the horizontal part
    img.putpixel((width - margin - 20, margin + 15), MIC)
    
    return img


def create_concert_hall(width: int = 200, height: int = 150) -> Image.Image:
    """Create a larger concert hall with absorbing walls."""
    img = Image.new('RGB', (width, height), WALL)
    draw = ImageDraw.Draw(img)
    
    margin = 5
    
    # Main hall
    draw.rectangle([margin, margin, width - margin - 1, height - margin - 1], fill=AIR)
    
    # Absorbing back wall (gray)
    draw.rectangle([margin, margin, width - margin - 1, margin + 3], fill=ABSORBING_WALL)
    
    # Source on stage
    img.putpixel((width // 2, height - margin - 20), SOURCE)
    
    # Microphone in audience
    img.putpixel((width // 2, margin + 30), MIC)
    
    return img


def create_recording_studio(width: int = 80, height: int = 60) -> Image.Image:
    """Create a small recording studio with absorbing walls."""
    img = Image.new('RGB', (width, height), ABSORBING_WALL)  # Absorbing walls
    draw = ImageDraw.Draw(img)
    
    # Air in center
    margin = 5
    draw.rectangle([margin, margin, width - margin - 1, height - margin - 1], fill=AIR)
    
    # Source in center-left
    img.putpixel((width // 3, height // 2), SOURCE)
    
    # Microphone in front of source
    img.putpixel((width // 3 + 10, height // 2), MIC)
    
    return img


def create_two_rooms(width: int = 160, height: int = 80) -> Image.Image:
    """Create two connected rooms with a doorway."""
    img = Image.new('RGB', (width, height), WALL)
    draw = ImageDraw.Draw(img)
    
    margin = 5
    wall_thickness = 3
    
    # Left room
    draw.rectangle([margin, margin, width // 2 - wall_thickness, height - margin - 1], fill=AIR)
    
    # Right room
    draw.rectangle([width // 2 + wall_thickness, margin, width - margin - 1, height - margin - 1], fill=AIR)
    
    # Doorway connecting rooms
    door_y_start = height // 2 - 10
    door_y_end = height // 2 + 10
    draw.rectangle([width // 2 - wall_thickness, door_y_start, width // 2 + wall_thickness, door_y_end], fill=AIR)
    
    # Source in left room
    img.putpixel((margin + 15, height // 2), SOURCE)
    
    # Microphone in right room
    img.putpixel((width - margin - 15, height // 2), MIC)
    
    return img


ROOM_TYPES = {
    'simple': create_simple_room,
    'l_shaped': create_l_shaped_room,
    'concert_hall': create_concert_hall,
    'studio': create_recording_studio,
    'two_rooms': create_two_rooms,
}


def main():
    parser = argparse.ArgumentParser(description='Create example room PNGs')
    parser.add_argument('room_type', choices=list(ROOM_TYPES.keys()),
                       help='Type of room to create')
    parser.add_argument('-o', '--output', default='rooms/room.png',
                       help='Output PNG path')
    parser.add_argument('--width', type=int, help='Room width in pixels')
    parser.add_argument('--height', type=int, help='Room height in pixels')
    parser.add_argument('--all', action='store_true',
                       help='Create all room types')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if args.all:
        # Create all room types
        for room_type, create_func in ROOM_TYPES.items():
            output_path = os.path.join(output_dir or 'rooms', f'{room_type}.png')
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            img = create_func()
            img.save(output_path)
            print(f"Created {output_path} ({img.width}x{img.height})")
    else:
        # Create single room
        create_func = ROOM_TYPES[args.room_type]
        
        kwargs = {}
        if args.width:
            kwargs['width'] = args.width
        if args.height:
            kwargs['height'] = args.height
        
        img = create_func(**kwargs)
        img.save(args.output)
        print(f"Created {args.output} ({img.width}x{img.height})")
        
        # Print room info
        print(f"\nRoom configuration:")
        print(f"  Black pixels: fully reflective walls")
        print(f"  Gray pixels: absorbing walls (brighter = more absorbing)")
        print(f"  White pixels: air")
        print(f"  Red pixel: source position")
        print(f"  Cyan pixel: microphone position")


if __name__ == '__main__':
    main()
