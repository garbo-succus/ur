"""Build the Ur die from the tokens project's beveled D4 geometry.

White circles repeat on the three faces meeting each of two marked corners.
Run with Python 3; no third-party packages are needed.
"""
import base64
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def mul(a, s):
    return tuple(x * s for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def unit(a):
    return mul(a, 1 / math.sqrt(dot(a, a)))


doc = json.loads((HERE / 'd4.gltf').read_text())
data = bytearray((HERE / 'd4.bin').read_bytes())
positions = list(struct.iter_unpack('<3f', data[:1296]))
indices = struct.unpack('<444H', data[3456:4344])
faces = []
circles = []
normals = []
circle_indices = []
for i in range(0, len(indices), 3):
    vertices = [positions[j] for j in indices[i:i+3]]
    a, b, c = vertices
    normal = cross(sub(b, a), sub(c, a))
    if math.sqrt(dot(normal, normal)) < 0.2:
        continue  # Bevel triangles, not the four main faces.
    normal = unit(normal)
    faces.append(normal)
    centroid = mul(add(add(a, b), c), 1/3)
    for corner in vertices:
        # Mark the upper corner and the lower right corner only.
        if not (corner[1] > 0.7 or corner[0] > 0.7):
            continue
        center = add(add(mul(corner, 0.70), mul(centroid, 0.30)), mul(normal, 0.001))
        u = unit(sub(b, a))
        v = cross(normal, u)
        start = len(circles)
        circles.append(center)
        for step in range(48):
            angle = step * math.tau / 48
            circles.append(add(center, add(mul(u, 0.075 * math.cos(angle)), mul(v, 0.075 * math.sin(angle)))))
        normals.extend([normal] * 49)
        for step in range(48):
            circle_indices.extend([start, start + 1 + step, start + 1 + (step + 1) % 48])


def accessor(values, components, fmt, component_type, kind, target):
    while len(data) % 4:
        data.append(0)
    offset = len(data)
    flat = [x for value in values for x in value] if components > 1 else values
    data.extend(struct.pack('<' + fmt * len(flat), *flat))
    view = len(doc['bufferViews'])
    doc['bufferViews'].append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(data)-offset, 'target': target})
    acc = {'bufferView': view, 'componentType': component_type, 'count': len(values), 'type': kind}
    if kind == 'VEC3':
        acc['min'] = [min(v[j] for v in values) for j in range(3)]
        acc['max'] = [max(v[j] for v in values) for j in range(3)]
    doc['accessors'].append(acc)
    return len(doc['accessors'])-1


pos = accessor(circles, 3, 'f', 5126, 'VEC3', 34962)
nor = accessor(normals, 3, 'f', 5126, 'VEC3', 34962)
ind = accessor(circle_indices, 1, 'H', 5123, 'SCALAR', 34963)
doc['meshes'][0]['primitives'].append({'attributes': {'POSITION': pos, 'NORMAL': nor}, 'indices': ind, 'material': 1})
doc['materials'] = [
    {'name': 'Black die', 'pbrMetallicRoughness': {'baseColorFactor': [0.009, 0.009, 0.009, 1], 'metallicFactor': 0, 'roughnessFactor': 0.38}},
    {'name': 'White corner circles', 'pbrMetallicRoughness': {'baseColorFactor': [1, 1, 1, 1], 'metallicFactor': 0, 'roughnessFactor': 0.55}},
]
for key in ('textures', 'images', 'samplers', 'extensionsUsed', 'extensionsRequired'):
    doc.pop(key, None)
doc['nodes'][0]['name'] = 'Ur die'
doc['meshes'][0]['name'] = 'Beveled D4 with two marked corners'
doc['asset']['generator'] = 'source/ur-dice/generate.py; geometry from tokens D4'
doc['buffers'] = [{'byteLength': len(data)}]
# Authored metre dimensions match the current release's original D4s.
doc['nodes'][0]['scale'] = [0.016, 0.016, 0.016]
doc['buffers'][0]['uri'] = 'data:application/octet-stream;base64,' + base64.b64encode(data).decode()
(ROOT / 'pieces/ur-dice.gltf').write_text(json.dumps(doc, separators=(',', ':')) + '\n')
assert len(faces) == 4 and len(circles) == 6 * 49
print('Wrote pieces/ur-dice.gltf: two marked corners, six circles.')
