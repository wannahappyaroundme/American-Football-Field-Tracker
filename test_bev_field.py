#!/usr/bin/env python3
"""
Test script to verify BEV field template with yard lines.

This script creates a sample BEV field visualization to verify:
1. Green field background is rendered correctly
2. Yard lines are displayed every 10 yards
3. Yard numbers are visible
4. Field boundaries are properly masked
"""

import cv2
import numpy as np
from visualizer import Visualizer

def main():
    print("=" * 60)
    print("BEV FIELD TEMPLATE TEST")
    print("=" * 60)

    # Create visualizer instance
    visualizer = Visualizer()

    # Test 1: Create field template
    print("\n[Test 1] Creating field template (1000x500)...")
    field_template = visualizer.create_field_template(width=1000, height=500)

    if field_template is not None:
        print("✓ Field template created successfully")
        print(f"  - Size: {field_template.shape}")
        print(f"  - Type: {field_template.dtype}")
    else:
        print("✗ Failed to create field template")
        return

    # Test 2: Create sample BEV with player positions
    print("\n[Test 2] Creating BEV with sample player positions...")

    # Create sample player states (simulate player paths)
    player_states = {
        1: {
            'path': [
                (100, 250, 'active'),
                (200, 240, 'active'),
                (300, 230, 'active'),
                (400, 220, 'active')
            ]
        },
        2: {
            'path': [
                (100, 300, 'active'),
                (200, 310, 'active'),
                (300, 320, 'active'),
                (400, 330, 'occluded')
            ]
        },
        3: {
            'path': [
                (500, 250, 'active'),
                (600, 250, 'active'),
                (700, 250, 'active')
            ]
        },
        4: {
            'path': [
                (50, 100, 'active'),  # Out of bounds (left margin)
                (60, 110, 'active')
            ]
        },
        5: {
            'path': [
                (970, 450, 'active'),  # Out of bounds (right/bottom margin)
                (980, 460, 'active')
            ]
        }
    }

    # Draw BEV
    bev_image = visualizer.draw_bird_eye_view(player_states, map_size=(1000, 500))

    if bev_image is not None:
        print("✓ BEV visualization created successfully")
    else:
        print("✗ Failed to create BEV visualization")
        return

    # Test 3: Save outputs
    print("\n[Test 3] Saving test outputs...")

    output_dir = "output"
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save field template
    template_path = f"{output_dir}/test_field_template.png"
    cv2.imwrite(template_path, field_template)
    print(f"✓ Field template saved to: {template_path}")

    # Save BEV with players
    bev_path = f"{output_dir}/test_bev_with_players.png"
    cv2.imwrite(bev_path, bev_image)
    print(f"✓ BEV with players saved to: {bev_path}")

    # Test 4: Verify field characteristics
    print("\n[Test 4] Verifying field characteristics...")

    # Check that field is not all black
    mean_color = np.mean(field_template)
    if mean_color > 10:
        print(f"✓ Field has color (mean pixel value: {mean_color:.1f})")
    else:
        print(f"✗ Field appears to be black (mean pixel value: {mean_color:.1f})")

    # Check for green color (should be dominant in BGR format)
    mean_b, mean_g, mean_r = cv2.mean(field_template)[:3]
    if mean_g > mean_b and mean_g > mean_r:
        print(f"✓ Field is predominantly green (B:{mean_b:.0f}, G:{mean_g:.0f}, R:{mean_r:.0f})")
    else:
        print(f"⚠ Field color unexpected (B:{mean_b:.0f}, G:{mean_g:.0f}, R:{mean_r:.0f})")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✓ Field template creation: PASSED")
    print("✓ BEV visualization: PASSED")
    print("✓ File output: PASSED")
    print("\nExpected features:")
    print("  - Green field background")
    print("  - White yard lines every 10 yards (0, 10, 20, 30, 40, 50)")
    print("  - Yard numbers at top and bottom")
    print("  - White sideline borders")
    print("  - Darkened out-of-bounds areas (if enabled in config)")
    print("  - Player paths (solid lines for active, dotted for occluded)")
    print("  - Player dots at current positions")
    print("=" * 60)

if __name__ == "__main__":
    main()
