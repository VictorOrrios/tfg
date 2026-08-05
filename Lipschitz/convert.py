#!/usr/bin/env python3

import json
import math
import itertools
import numpy as np

PRIM = {
    "empty": 0, "box": 1, "sphere": 2, "torus": 3,
    "snowman": 4, "plane": 5, "cylinder": 6, "cone": 7,
}

COMB = {"union": 0, "sub": 1}

next_id = itertools.count(1)

def load_matrix(m):
    return np.array([
        [m[0], m[1], m[2],  m[3]],
        [m[4], m[5], m[6],  m[7]],
        [m[8], m[9], m[10], m[11]],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=float)

def matrix_to_quaternion(R):
    t = np.trace(R)

    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2,1] - R[1,2]) / s
        qy = (R[0,2] - R[2,0]) / s
        qz = (R[1,0] - R[0,1]) / s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
        qw = (R[2,1]-R[1,2])/s
        qx = 0.25*s
        qy = (R[0,1]+R[1,0])/s
        qz = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
        qw = (R[0,2]-R[2,0])/s
        qx = (R[0,1]+R[1,0])/s
        qy = 0.25*s
        qz = (R[1,2]+R[2,1])/s
    else:
        s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
        qw = (R[1,0]-R[0,1])/s
        qx = (R[0,2]+R[2,0])/s
        qy = (R[1,2]+R[2,1])/s
        qz = 0.25*s

    return [qx, qy, qz, qw]

def make_node():
    return {
        "id": next(next_id),
        "type": 0,
        "material": 0,
        "position": [0,0,0],
        "rotation": [0,0,0,1],
        "scale": 1.0,
        "tInv": np.identity(4),
        "bbox": [[-.5,-.5,-.5],[.5,.5,.5]],
        "combOp": 0,
        "smoothness": 0.0,
        "primMod": [0.5,0.5,0.5,0.0],
    }

def decompose_matrix(M):
    # M is world-to-object. Extract object-to-world for position/rotation.
    tInv = M.copy()
    M_obj2world = np.linalg.inv(M)
    position = M_obj2world[:3, 3].copy()
    R_obj2world = M_obj2world[:3, :3]
    scale = np.linalg.norm(R_obj2world[:, 0])
    R_obj2world = R_obj2world / np.linalg.norm(R_obj2world, axis=0)
    rotation = matrix_to_quaternion(R_obj2world)
    return position, rotation, scale, tInv

def convert_primitive(j, M, comb, smooth):
    node = make_node()
    typ = j["primitiveType"]
    node["type"] = PRIM[typ]
    node["combOp"] = comb
    node["smoothness"] = smooth

    position, rotation, scale, tInv = decompose_matrix(M)
    node["position"] = position.tolist()
    node["rotation"] = rotation
    node["scale"] = scale
    node["tInv"] = tInv

    node["bbox"] = [[-.5,-.5,-.5],[.5,.5,.5]]
    node["primMod"] = [0.5,0.5,0.5,0]
    node["bevel"] = [0.0,0.0,0.0,0.0]
    node["round"] = [0.0,0.0]

    if typ == "sphere":
        node["primMod"] = [j["radius"], 0, 0, 0]
    elif typ == "box":
        sx, sy, sz = j["sides"]
        node["primMod"] = [sx*0.5, sy*0.5, sz*0.5, 0.0]
        bevel = j.get("bevel", [0.0, 0.0, 0.0, 0.0])
        node["bevel"] = [b * 0.5 for b in bevel]
        node["round"] = [j.get("round_x", 0.0), j.get("round_y", 0.0)]
    elif typ == "cylinder":
        node["primMod"] = [j["radius"], j["height"], 0, 0]
    elif typ == "cone":
        node["primMod"] = [j["radius"], j["height"], 0, 0]

    return node

def collect_subtraction_objects(scene):
    """
    Collect all primitives used as subtraction (right child of a sub operation).
    Groups them by their geometric parameters (ignoring transform matrix).
    Returns a dict with counts of each unique shape.
    """
    negative_objects = {}
    
    def traverse(node, parentMatrix=np.identity(4), is_subtraction=False):
        if node["nodeType"] == "primitive":
            if is_subtraction:
                # Create a signature based on primitive type and parameters
                M = parentMatrix @ load_matrix(node["matrix"])
                
                typ = node["primitiveType"]
                sig = {"type": typ}
                
                if typ == "sphere":
                    sig["radius"] = node.get("radius", 0)
                elif typ == "box":
                    sig["sides"] = tuple(node.get("sides", [0,0,0]))
                    sig["bevel"] = tuple(node.get("bevel", [0,0,0,0]))
                    sig["round_x"] = node.get("round_x", 0)
                    sig["round_y"] = node.get("round_y", 0)
                elif typ == "cylinder":
                    sig["radius"] = node.get("radius", 0)
                    sig["height"] = node.get("height", 0)
                elif typ == "cone":
                    sig["radius"] = node.get("radius", 0)
                    sig["height"] = node.get("height", 0)
                elif typ == "torus":
                    sig["torus_radius"] = node.get("torus_radius", 0)
                    sig["tube_radius"] = node.get("tube_radius", 0)
                
                # Create a hashable key from the signature
                key = str(sig)
                
                if key not in negative_objects:
                    negative_objects[key] = {
                        "signature": sig,
                        "count": 0,
                        "examples": []
                    }
                
                negative_objects[key]["count"] += 1
                # Store first 3 examples
                if len(negative_objects[key]["examples"]) < 3:
                    negative_objects[key]["examples"].append({
                        "position": M[:3, 3].tolist(),
                        "scale": np.linalg.norm(M[:3, 0])
                    })
            return
        
        mode = node["blendMode"]
        childMatrix = parentMatrix @ load_matrix(node["matrix"])
        
        if mode == "sub":
            # Left child: not subtraction (what we subtract FROM)
            traverse(node["leftChild"], childMatrix, False)
            # Right child: IS subtraction (what we subtract WITH)
            traverse(node["rightChild"], childMatrix, True)
        else:
            traverse(node["leftChild"], childMatrix, is_subtraction)
            traverse(node["rightChild"], childMatrix, is_subtraction)
    
    traverse(scene)
    return negative_objects

def print_subtraction_analysis(scene):
    """Print analysis of subtraction objects."""
    neg_objs = collect_subtraction_objects(scene)
    
    print(f"\n=== SUBTRACTION OBJECTS ANALYSIS ===")
    print(f"Total unique subtraction shapes: {len(neg_objs)}")
    print(f"Total subtraction primitives: {sum(v['count'] for v in neg_objs.values())}")
    
    # Sort by count (most frequent first)
    sorted_objs = sorted(neg_objs.values(), key=lambda x: x['count'], reverse=True)
    
    print("\nSubtraction shapes by frequency:")
    for i, obj in enumerate(sorted_objs):
        sig = obj["signature"]
        count = obj["count"]
        
        print(f"\n  Shape #{i+1}: {sig['type']} (used {count} times)")
        
        if sig['type'] == "box":
            print(f"    sides: {sig['sides']}")
            print(f"    bevel: {sig['bevel']}")
            print(f"    round_x: {sig['round_x']}")
            print(f"    round_y: {sig['round_y']}")
        elif sig['type'] in ("sphere", "cylinder", "cone"):
            for k, v in sig.items():
                if k != 'type':
                    print(f"    {k}: {v}")
        
        # Show first example position
        if obj["examples"]:
            ex = obj["examples"][0]
            print(f"    example position: {ex['position']}")
            if len(obj["examples"]) > 1:
                print(f"    ({len(obj['examples'])} examples stored)")
    
    return neg_objs

nodes = []

def visit(node, parentMatrix=np.identity(4), comb=0, smooth=0.0):
    if node["nodeType"] == "primitive":
        if smooth > 0.0 and comb == 1:
            comb = 3
        elif smooth > 0.0 and comb == 0:
            comb = 2
        M = parentMatrix @ load_matrix(node["matrix"])
        nodes.append(convert_primitive(node, M, comb, smooth * 0.25))
        return

    childMatrix = parentMatrix @ load_matrix(node["matrix"])
    mode = node["blendMode"]
    blend = node.get("blendRadius", 0.0)

    if mode == "sub":
        # Left child: always union (it's what we're subtracting FROM)
        visit(node["leftChild"], childMatrix, 0, blend)
        # Right child: always sub (it's what we're subtracting WITH)
        visit(node["rightChild"], childMatrix, 1, blend)
    else:  # union
        visit(node["leftChild"], childMatrix, 0, blend)
        visit(node["rightChild"], childMatrix, 0, blend)

def vec3(v):
    return {"value0": float(v[0]), "value1": float(v[1]), "value2": float(v[2])}

def ivec3(v):
    return {"value0": int(v[0]), "value1": int(v[1]), "value2": int(v[2])}

def vec4(v):
    return {"value0": float(v[0]), "value1": float(v[1]), "value2": float(v[2]), "value3": float(v[3])}

def vec2(v):
    return {"value0": float(v[0]), "value1": float(v[1])}

# Serialize matrix in column-major order for GLM
def mat4(M):
    return {
        "value0": float(M[0,0]), "value1": float(M[1,0]), "value2": float(M[2,0]), "value3": float(M[3,0]),
        "value4": float(M[0,1]), "value5": float(M[1,1]), "value6": float(M[2,1]), "value7": float(M[3,1]),
        "value8": float(M[0,2]), "value9": float(M[1,2]), "value10": float(M[2,2]), "value11": float(M[3,2]),
        "value12": float(M[0,3]), "value13": float(M[1,3]), "value14": float(M[2,3]), "value15": float(M[3,3]),
    }

def bbox(b):
    return {"value0": vec3(b[0]), "value1": vec3(b[1])}

def physics():
    return {
        "value0": False, "value1": 0.0, "value2": 0.0,
        "value3": vec3([0,0,0]), "value4": vec4([0,0,0,1]), "value5": vec4([0,0,0,1]),
        "value6": vec3([0,0,0]), "value7": vec3([0,0,0]), "value8": vec3([0,0,0]),
        "value9": vec3([0,0,0]), "value10": vec3([0,0,0]), "value11": vec3([0,0,0]),
    }

def guizmo():
    return {"value0": 7, "value1": 1, "value2": mat4(np.identity(4))}

def sdf(node):
    return {
        "value0": 0.0, "value1": node["combOp"], "value2": node["smoothness"],
        "value3": 0, "value4": vec3([0,0,0]), "value5": ivec3([0,0,0]),
        "value6": 0, "value7": vec3([0,0,0]), "value8": 0,
        "value9": vec4([0,0,0,0]), "value10": 0, "value11": 0.0,
        "value12": vec4(node["primMod"]),
        "value13": vec4(node["bevel"]),
        "value14": vec2(node["round"])
    }

def general(node):
    return {
        "value0": node["type"], "value1": 0,
        "value2": vec3(node["position"]), "value3": vec4(node["rotation"]),
        "value4": mat4(node["tInv"]), "value5": node["scale"],
        "value6": bbox(node["bbox"]), "value7": bbox(node["bbox"])
    }

def cereal_node(node):
    return {
        "value0": node["id"], "value1": False, "value2": False,
        "value3": general(node), "value4": sdf(node),
        "value5": physics(), "value6": guizmo(),
    }

materials = [{
    "value0": 1, "value1": "Default",
    "value2": vec3([1,1,1]), "value3": 64.0,
    "value4": 0.8, "value5": 0.0, "value6": 0,
}]



def save_scene(path):
    out = {
        "value0": [cereal_node(n) for n in nodes],
        "value1": materials
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=4)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("usage: python convert.py input.json output.json")
        quit()

    scene = json.load(open(sys.argv[1]))
    
    print_subtraction_analysis(scene)

    visit(scene)
    print(len(nodes), "nodes")
    save_scene(sys.argv[2])