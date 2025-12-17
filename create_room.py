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
WALL = (0, 0, 0)
AIR = (255, 255, 255)
SOURCE_RED = (255, 0, 0)
SOURCE_GREEN = (0, 255, 0)
SOURCE_BLUE = (0, 0, 255)
MIC_CYAN = (0, 255, 255)
MIC_MAGENTA = (255, 0, 255)
ABSORBING_GRAY = (128, 128, 128)  # 50% absorption


def create_simple_room(width: int = 100, height: int = 80) -> Image.Image:
    """Create a simple rectangular room with one source and one microphone."""
    img = Image.new('RGB', (width, height), WALL)
    draw = ImageDraw.Draw(img)
    
    # Inner room (air)
    margin = 5
    draw.rectangle([margin, margin, width - margin - 1, height - margin - 1], fill=AIR)
    
    # Source in upper-left area (red)
    img.putpixel((margin + 10, height - margin - 15), SOURCE_RED)
    
    # Microphone in lower-right area (cyan)
    img.putpixel((width - margin - 15, margin + 10), MIC_CYAN)
    
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
    
    # Source in the vertical part (red)
    img.putpixel((margin + 15, height - margin - 20), SOURCE_RED)
    
    # Microphone in the horizontal part (cyan)
    img.putpixel((width - margin - 20, margin + 15), MIC_CYAN)
    
    return img


def create_concert_hall(width: int = 200, height: int = 150) -> Image.Image:
    """Create a larger concert hall with stage area and multiple sources/mics."""
    img = Image.new('RGB', (width, height), WALL)
    draw = ImageDraw.Draw(img)
    
    margin = 5
    
    # Main hall
    draw.rectangle([margin, margin, width - margin - 1, height - margin - 1], fill=AIR)
    
    # Stage area (slightly absorbing - gray)
    stage_height = 30
    draw.rectangle([margin, height - margin - stage_height, width - margin - 1, height - margin - 1], 
                   fill=ABSORBING_GRAY)
    
    # Center source on stage (red - main performer)
    img.putpixel((width // 2, height - margin - stage_height // 2), SOURCE_RED)
    
    # Left source (green - left speaker)
    img.putpixel((margin + 20, height - margin - stage_height // 2), SOURCE_GREEN)
    
    # Right source (blue - right speaker)
    img.putpixel((width - margin - 20, height - margin - stage_height // 2), SOURCE_BLUE)
    
    # Audience microphones
    # Front row center (cyan)
    img.putpixel((width // 2, height - margin - stage_height - 20), MIC_CYAN)
    
    # Back row center (magenta)
    img.putpixel((width // 2, margin + 20), MIC_MAGENTA)
    
    return img


def create_recording_studio(width: int = 80, height: int = 60) -> Image.Image:
    """Create a small recording studio with absorbing walls."""
    img = Image.new('RGB', (width, height), WALL)
    draw = ImageDraw.Draw(img)
    
    margin = 3
    
    # Inner room with absorbing material (gray walls)
    draw.rectangle([margin, margin, width - margin - 1, height - margin - 1], fill=ABSORBING_GRAY)
    
    # Air in center (less absorption)
    inner_margin = 8
    draw.rectangle([inner_margin, inner_margin, width - inner_margin - 1, height - inner_margin - 1], fill=AIR)
    
    # Singer/source in center-left (red)
    img.putpixel((width // 3, height // 2), SOURCE_RED)
    
    # Microphone in front of source (cyan)
    img.putpixel((width // 3 + 10, height // 2), MIC_CYAN)
    
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
    
    # Source in left room (red)
    img.putpixel((margin + 15, height // 2), SOURCE_RED)
    
    # Microphone in right room (cyan)
    img.putpixel((width - margin - 15, height // 2), MIC_CYAN)
    
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
        print(f"  Black pixels: walls")
        print(f"  White pixels: air")
        print(f"  Gray pixels: absorbing material")
        print(f"  Red pixel: source (use with --sources voice.wav)")
        print(f"  Green pixel: source 2")
        print(f"  Blue pixel: source 3")
        print(f"  Cyan pixel: microphone")
        print(f"  Magenta pixel: microphone 2")


if __name__ == '__main__':
    main()
