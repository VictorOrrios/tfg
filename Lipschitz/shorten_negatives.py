#!/usr/bin/env python3

import json
import math
import itertools
import numpy as np
def rewrite_all_subtractions(input_path, output_path):
    """
    Rewrite the JSON file, modifying all subtraction primitives.
    Define new sizes in the SUBSTITUTION_TABLE below.
    """
    with open(input_path, 'r') as f:
        scene = json.load(f)
    
    # ============================================================
    # SUBSTITUTION TABLE: Edit this to change sizes
    # Format: (type, sides, bevel, round_x, round_y) -> new_sides
    # Use None for fields you don't want to match on
    # ============================================================
    
    SUBSTITUTION_TABLE = [
        # (original_sides, original_bevel, original_round_x, original_round_y) -> new_sides
        # Shape #1: 48 times
        {
            "match_sides": (1.0, 0.04, 0.02),
            "match_bevel": (0.0, 0.0, 0.0, 0.0),
            "match_round_x": 0.02,
            "match_round_y": 0.0,
            "new_sides": [0.0, 0.04, 0.02],  # Shorter and thinner
        },
        # Shape #2: 48 times
        {
            "match_sides": (1.0, 1.0, 0.02),
            "match_bevel": (0.0, 0.0, 0.0, 0.0),
            "match_round_x": 0.0,
            "match_round_y": 0.0,
            "new_sides": [0.0, 0.0, 0.02],  # Thinner
        },
        # Shape #3: 16 times
        {
            "match_sides": (1.0, 0.06, 0.03),
            "match_bevel": (0.0, 0.0, 0.0, 0.0),
            "match_round_x": 0.03,
            "match_round_y": 0.0,
            "new_sides": [0.0, 0.06, 0.03],  # Shorter
        },
        # Shape #4: 16 times
        {
            "match_sides": (1.0, 1.0, 0.03),
            "match_bevel": (0.0, 0.0, 0.0, 0.0),
            "match_round_x": 0.0,
            "match_round_y": 0.0,
            "new_sides": [0.0, 0.0, 0.03],  # Thinner
        },
        # LOWER ARCHES
        # Shape #5: 16 times (has bevel)
        {
            "match_sides": (0.065, 1.32, 0.105),
            "match_bevel": (0.0, 0.065, 0.0, 0.065),
            "match_round_x": 0.0,
            "match_round_y": 0.0,
            "new_sides": [0.065, 1.22, 0.105],  # Thinner
        },
    ]
    
    modified_counts = {}
    
    def signaturas_match(node, entry):
        """Check if a primitive matches a table entry."""
        if node.get("primitiveType") != "box":
            return False
        
        sides = tuple(node.get("sides", []))
        bevel = tuple(node.get("bevel", [0,0,0,0]))
        round_x = node.get("round_x", 0.0)
        round_y = node.get("round_y", 0.0)
        
        tol = 0.001  # Relaxed tolerance for matching
        
        match_sides = tuple(entry["match_sides"])
        match_bevel = tuple(entry["match_bevel"])
        
        if len(sides) != 3:
            return False
        
        if (abs(sides[0] - match_sides[0]) > tol or
            abs(sides[1] - match_sides[1]) > tol or
            abs(sides[2] - match_sides[2]) > tol):
            return False
        
        if (abs(bevel[0] - match_bevel[0]) > tol or
            abs(bevel[1] - match_bevel[1]) > tol or
            abs(bevel[2] - match_bevel[2]) > tol or
            abs(bevel[3] - match_bevel[3]) > tol):
            return False
        
        if abs(round_x - entry["match_round_x"]) > tol:
            return False
        
        if abs(round_y - entry["match_round_y"]) > tol:
            return False
        
        return True
    
    def traverse_and_modify(node, is_subtraction=False):
        nonlocal modified_counts
        
        if node["nodeType"] == "primitive":
            if is_subtraction and node.get("primitiveType") == "box":
                for i, entry in enumerate(SUBSTITUTION_TABLE):
                    if signaturas_match(node, entry):
                        old_sides = node["sides"].copy()
                        node["sides"] = entry["new_sides"]
                        
                        key = f"Shape #{i+1}"
                        if key not in modified_counts:
                            modified_counts[key] = 0
                        modified_counts[key] += 1
                        
                        print(f"  {key}: {old_sides} -> {entry['new_sides']}")
                        break
            return
        
        mode = node.get("blendMode", "union")
        
        if mode == "sub":
            traverse_and_modify(node["leftChild"], False)
            traverse_and_modify(node["rightChild"], True)
        else:
            traverse_and_modify(node["leftChild"], is_subtraction)
            traverse_and_modify(node["rightChild"], is_subtraction)
    
    print("Modifying subtraction primitives...")
    traverse_and_modify(scene)
    
    print(f"\nModified {sum(modified_counts.values())} primitives total")
    for key, count in modified_counts.items():
        print(f"  {key}: {count} times")
    
    with open(output_path, 'w') as f:
        json.dump(scene, f, indent=2)
    
    return sum(modified_counts.values())

# Usage:
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("usage: python convert.py input.json output.json")
        quit()
    
    rewrite_all_subtractions(sys.argv[1], sys.argv[2])