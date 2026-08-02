#!/usr/bin/env python3

import json
import math
import itertools
import numpy as np

#------------------------------------------------------------
# Enumeraciones de tu motor
#------------------------------------------------------------

PRIM = {
    "empty":      0,
    "box":        1,
    "sphere":     2,
    "torus":      3,
    "snowman":    4,
    "plane":      5,
    "cylinder":   6,
    "cone":       7,
}

COMB = {
    "union": 0,
    "sub":   1,
}

#------------------------------------------------------------
# Ids
#------------------------------------------------------------

next_id = itertools.count(1)

#------------------------------------------------------------
# Matrices
#------------------------------------------------------------

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

    return [qx,qy,qz,qw]


#------------------------------------------------------------
# AABB local
#------------------------------------------------------------

def primitive_bbox(typ, prim):

    if typ=="sphere":

        r=prim["radius"]
        return [-r,-r,-r],[r,r,r]

    elif typ=="box":

        sx,sy,sz=prim["sides"]

        b=np.array([sx,sy,sz])*0.5

        bevel=max(prim["bevel"])

        return (b*(-1)-bevel).tolist(),(b+bevel).tolist()

    elif typ=="cylinder":

        r=prim["radius"]
        h=prim["height"]*0.5

        return [-r,-h,-r],[r,h,r]

    elif typ=="cone":

        r=prim["radius"]
        h=prim["height"]*0.5

        return [-r,-h,-r],[r,h,r]

    return [-.5,-.5,-.5],[.5,.5,.5]


#------------------------------------------------------------
# Nodo intermedio
#------------------------------------------------------------

def make_node():

    return {

        "id": next(next_id),

        "type":0,

        "material":0,

        "position":[0,0,0],

        "rotation":[0,0,0,1],

        "scale":1.0,

        "tInv":np.identity(4),

        "bbox":[[-.5,-.5,-.5],[.5,.5,.5]],

        "combOp":0,

        "smoothness":0.0,

        "primMod":[0.5,0.5,0.5,0.0],

    }


#------------------------------------------------------------
# Conversión de una primitiva
#------------------------------------------------------------

def decompose_matrix(M):

    position = M[:3,3].copy()

    R = M[:3,:3]

    rotation = matrix_to_quaternion(R)

    TR = np.identity(4)
    TR[:3,:3] = R
    TR[:3,3] = position

    tInv = np.linalg.inv(TR)

    return position, rotation, 1.0, tInv

def convert_primitive(j,M,comb,smooth):

    node=make_node()

    typ=j["primitiveType"]

    node["type"]=PRIM[typ]

    node["combOp"]=comb

    node["smoothness"]=smooth

    position, rotation, scale, tInv = decompose_matrix(M)

    node["position"] = position.tolist()
    node["rotation"] = rotation
    node["scale"]    = scale
    node["tInv"]     = tInv

    mn,mx=primitive_bbox(typ,j)

    node["bbox"]=[mn,mx]

    if typ=="sphere":

        node["primMod"]=[
            j["radius"],
            0,
            0,
            0
        ]

    elif typ=="box":

        sx,sy,sz=j["sides"]

        node["primMod"]=[
            sx*0.5,
            sy*0.5,
            sz*0.5,
            max(j["bevel"]) * 0.25
        ]

    elif typ=="cylinder":

        node["primMod"]=[
            j["radius"],
            j["height"],
            0,
            0
        ]

    elif typ=="cone":

        node["primMod"]=[
            j["radius"],
            j["height"],
            0,
            0
        ]

    return node


#------------------------------------------------------------
# Recorrido del árbol
#------------------------------------------------------------

nodes=[]

def visit(node, parentMatrix=np.identity(4), comb=0, smooth=0.0):

    if node["nodeType"] == "primitive":

        nodes.append(
            convert_primitive(
                node,
                parentMatrix @ load_matrix(node["matrix"]),
                comb,
                smooth * 0.25
            )
        )
        return

    childMatrix = parentMatrix @ load_matrix(node["matrix"])

    mode = node["blendMode"]

    if mode == "union":
        op = 0
    elif mode == "sub":
        op = 1
    else:
        print(f"WARNING: blendMode '{mode}' no soportado, usando Union")
        op = 0
    
    if smooth > 0.0:
        op += 2

    # El hijo izquierdo mantiene la operación heredada
    visit(
        node["leftChild"],
        childMatrix,
        comb,
        smooth
    )

    # El hijo derecho recibe la operación y el blend de ESTE operador
    visit(
        node["rightChild"],
        childMatrix,
        op,
        node.get("blendRadius", 0.0)
    )

#------------------------------------------------------------
# Utilidades JSON cereal
#------------------------------------------------------------

def vec3(v):
    return {
        "value0": float(v[0]),
        "value1": float(v[1]),
        "value2": float(v[2]),
    }


def ivec3(v):
    return {
        "value0": int(v[0]),
        "value1": int(v[1]),
        "value2": int(v[2]),
    }


def vec4(v):
    return {
        "value0": float(v[0]),
        "value1": float(v[1]),
        "value2": float(v[2]),
        "value3": float(v[3]),
    }


def mat4(M):

    return {

        "value0": float(M[0,0]),
        "value1": float(M[0,1]),
        "value2": float(M[0,2]),
        "value3": float(M[0,3]),

        "value4": float(M[1,0]),
        "value5": float(M[1,1]),
        "value6": float(M[1,2]),
        "value7": float(M[1,3]),

        "value8": float(M[2,0]),
        "value9": float(M[2,1]),
        "value10": float(M[2,2]),
        "value11": float(M[2,3]),

        "value12": float(M[3,0]),
        "value13": float(M[3,1]),
        "value14": float(M[3,2]),
        "value15": float(M[3,3]),
    }


def bbox(b):

    return {
        "value0": vec3(b[0]),
        "value1": vec3(b[1]),
    }

def physics():

    return {

        "value0": False,
        "value1": 0.0,
        "value2": 0.0,

        "value3": vec3([0,0,0]),
        "value4": vec4([0,0,0,1]),
        "value5": vec4([0,0,0,1]),

        "value6": vec3([0,0,0]),
        "value7": vec3([0,0,0]),
        "value8": vec3([0,0,0]),

        "value9": vec3([0,0,0]),
        "value10": vec3([0,0,0]),
        "value11": vec3([0,0,0]),
    }

def guizmo():

    return {

        "value0":7,
        "value1":1,
        "value2":mat4(np.identity(4))

    }

def sdf(node):

    return {

        "value0":0.0,                         # roundness
        "value1":node["combOp"],
        "value2":node["smoothness"],

        "value3":0,                           # repetition

        "value4":vec3([0,0,0]),

        "value5":ivec3([0,0,0]),

        "value6":0,                           # deformation

        "value7":vec3([0,0,0]),

        "value8":0,                          # octaves

        "value9":vec4([0,0,0,0]),             # terrain

        "value10":0,
        "value11":0.0,

        "value12":vec4(node["primMod"])
    }

def general(node):

    return {

        "value0":node["type"],

        "value1":0,                      # material

        "value2":vec3(node["position"]),

        "value3":vec4(node["rotation"]),

        "value4":mat4(node["tInv"]),

        "value5":node["scale"],

        "value6":bbox(node["bbox"]),

        "value7":bbox(node["bbox"])
    }

def cereal_node(node):

    return {

        "value0":node["id"],

        "value1":False,

        "value2":False,

        "value3":general(node),

        "value4":sdf(node),

        "value5":physics(),

        "value6":guizmo(),

    }

materials=[

{
    "value0":1,
    "value1":"Default",

    "value2":vec3([1,1,1]),

    "value3":64.0,

    "value4":0.8,

    "value5":0.0,

    "value6":0,
}

]

def save_scene(path):

    out={

        "value0":[
            cereal_node(n)
            for n in nodes
        ],

        "value1":materials

    }

    with open(path,"w") as f:
        json.dump(out,f,indent=4)


if __name__=="__main__":

    import sys

    if len(sys.argv)!=3:
        print("usage:")
        print("python convert.py input.json output.json")
        quit()

    scene=json.load(open(sys.argv[1]))

    base_matrix = np.array([
        [1, 0,  0, 0],
        [0, -1, 0, 0],
        [0, 0,  1, 0],
        [0, 0,  0, 1]
    ], dtype=float)

    visit(scene, base_matrix)

    print(len(nodes),"nodes")

    save_scene(sys.argv[2])