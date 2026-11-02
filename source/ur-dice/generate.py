"""Generate a regular, centred Ur tetrahedron with uniform rounded edges.

Stdlib only. Coordinates, normals, UVs, markings and roll poses are derived
from the same mathematical solid; no imported mesh transforms are needed.
"""
import base64
from itertools import combinations
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
RADIUS = 1.125
SCALE = 0.016
BEVEL = 0.0012 / SCALE
SEGMENTS = 2
PIP_SEGMENTS = 16
TILT = math.acos(-1 / 3)
TURN = math.pi / 3


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


# Top, lower right, lower back, lower left. Every pair has the same distance.
vertices = [mul(v, RADIUS) for v in (
    (0, 1, 0),
    (math.sqrt(2/3), -1/3, math.sqrt(2)/3),
    (0, -1/3, -2*math.sqrt(2)/3),
    (-math.sqrt(2/3), -1/3, math.sqrt(2)/3),
)]
face_normals = [mul(unit(v), -1) for v in vertices]
# Minkowski sum of a smaller regular tetrahedron and a sphere: planar faces,
# cylindrical edge strips and spherical corners all meet tangentially.
inner = [mul(v, 1 - 3*BEVEL/RADIUS) for v in vertices]
positions, normals, uvs, indices = [], [], [], []


def vertex(point, normal, uv, patch):
    index = len(positions)
    positions.append(point)
    normals.append(normal)
    # Four face islands include their rounded rims in a spacious 2x2 grid.
    uvs.append(((patch % 2 + 0.08 + 0.84*uv[0])/2,
                (patch // 2 + 0.04 + 0.84*uv[1])/2))
    return index


def triangle(a, b, c):
    normal = cross(sub(positions[b], positions[a]), sub(positions[c], positions[a]))
    if dot(normal, add(add(normals[a], normals[b]), normals[c])) < 0:
        b, c = c, b
    indices.extend((a, b, c))


face_points = []
frames = []
for opposite, normal in enumerate(face_normals):
    corners = [i for i in range(4) if i != opposite]
    points = [add(inner[i], mul(normal, BEVEL)) for i in corners]
    face_points.append(dict(zip(corners, points)))
    centre = mul(normal, RADIUS/3)
    # Stable planar projection shared by the face and its curved rim.
    up = unit(sub(vertices[corners[0]], centre))
    right = cross(up, normal)
    radius = math.dist(vertices[corners[0]], centre)
    frames.append((centre, right, up, radius))


def face_vertex(point, normal, face):
    centre, u, v, radius = frames[face]
    delta = sub(point, centre)
    return vertex(point, normal, (0.5+dot(delta,u)/(2*radius),
                                  0.5+dot(delta,v)/(2*radius)), face)


for face, corners in enumerate(face_points):
    triangle(*(face_vertex(p, face_normals[face], face) for p in corners.values()))
for a, b in combinations(range(4),2):
    neighbours = [i for i in range(4) if i not in (a,b)]
    middle = unit(add(*(face_normals[i] for i in neighbours)))
    for face in neighbours:
        rows=[]
        for step in range(SEGMENTS+1):
            t=step/SEGMENTS
            n=unit(add(mul(face_normals[face],1-t),mul(middle,t)))
            rows.append([face_vertex(add(inner[i],mul(n,BEVEL)),n,face) for i in (a,b)])
        for p,q in zip(rows,rows[1:]):
            triangle(p[0],p[1],q[1]); triangle(p[0],q[1],q[0])
for corner in range(4):
    neighbours=[i for i in range(4) if i != corner]
    centre=unit(tuple(sum(face_normals[i][k] for i in neighbours) for k in range(3)))
    for a,b in combinations(neighbours,2):
        middle=unit(add(face_normals[a],face_normals[b]))
        for face in (a,b):
            triangle_normals=[centre,face_normals[face],middle]
            grid={}
            for i in range(SEGMENTS+1):
                for j in range(SEGMENTS+1-i):
                    weights=((SEGMENTS-i-j)/SEGMENTS,i/SEGMENTS,j/SEGMENTS)
                    n=unit(tuple(sum(n[k]*w for n,w in zip(triangle_normals,weights)) for k in range(3)))
                    grid[i,j]=face_vertex(add(inner[corner],mul(n,BEVEL)),n,face)
            for i in range(SEGMENTS):
                for j in range(SEGMENTS-i):
                    triangle(grid[i,j],grid[i+1,j],grid[i,j+1])
                    if i+j<SEGMENTS-1:
                        triangle(grid[i+1,j],grid[i+1,j+1],grid[i,j+1])

body = (positions, normals, uvs, indices)
positions, normals, uvs, indices = [], [], [], []
for face, corners in enumerate(face_points):
    normal = face_normals[face]
    centroid = mul(tuple(sum(p[i] for p in corners.values()) for i in range(3)), 1/3)
    points = list(corners.values())
    u = unit(sub(points[1], points[0]))
    v = cross(normal, u)
    for corner, point in corners.items():
        if corner not in (0, 1):
            continue
        center = add(add(mul(point, 0.70), mul(centroid, 0.30)), mul(normal, 0.001))
        start = len(positions)
        island = start // (PIP_SEGMENTS + 1)

        def pip_uv(x, y):
            # Separate islands allow future texture edits without repeating
            # a mark on all six circles. Gutters prevent filtering bleed.
            return ((island % 3 + 0.04 + 0.92*x)/3,
                    (island // 3 + 0.04 + 0.92*y)/2)

        positions.append(center)
        normals.append(normal)
        uvs.append(pip_uv(0.5, 0.5))
        for step in range(PIP_SEGMENTS):
            angle = step * math.tau / PIP_SEGMENTS
            positions.append(add(center, add(mul(u, 0.075*math.cos(angle)), mul(v, 0.075*math.sin(angle)))))
            normals.append(normal)
            uvs.append(pip_uv(0.5+0.5*math.cos(angle), 0.5+0.5*math.sin(angle)))
        for step in range(PIP_SEGMENTS):
            triangle(start, start+1+step, start+1+(step+1) % PIP_SEGMENTS)
pips = (positions, normals, uvs, indices)
assert len(positions) == 6*(PIP_SEGMENTS + 1)

# Each face pose brings the opposite corner to the top. The top and right
# corners are marked, preserving the existing outcome order 0, 0, 1, 1.
# These are symmetries of the whole tetrahedron, so every outcome has the
# same footprint and silhouette. Merely aligning a face normal leaves a
# free table-axis turn: Rx(TILT) alone points the footprint the other way.
# Compose a world Y half-turn with those poses, simplified in XYZ order:
# Ry(pi) Rx(TILT) Ry(theta) = Rx(-TILT) Ry(pi + theta).
faces = [{'name': name, 'rotation': rotation} for name, rotation in (
    ('0', [-TILT, math.pi, 0]), ('0', [-TILT, TURN, 0]),
    ('1', [-TILT, -TURN, 0]), ('1', [0, 0, 0]),
)]
doc = {
    'asset': {'version': '2.0', 'generator': 'source/ur-dice/generate.py; regular rounded tetrahedron'},
    'scene': 0, 'scenes': [{'nodes': [0]}],
    'nodes': [{'mesh': 0, 'name': 'Ur die', 'scale': [SCALE]*3}],
    'meshes': [{'name': 'Regular Ur tetrahedron', 'primitives': []}],
    'materials': [
        {'name': 'Black die', 'pbrMetallicRoughness': {'baseColorFactor': [0.009, 0.009, 0.009, 1], 'metallicFactor': 0, 'roughnessFactor': 0.38}},
        {'name': 'White corner circles', 'pbrMetallicRoughness': {'baseColorFactor': [1, 1, 1, 1], 'metallicFactor': 0, 'roughnessFactor': 0.55}},
    ],
    'bufferViews': [], 'accessors': [],
    'extras': {'probability': {'faces': faces}},
}
data = bytearray()


def accessor(values, components, kind, fmt='f', component_type=5126):
    while len(data) % 4:
        data.append(0)
    offset = len(data)
    flat = [v for row in values for v in row] if components > 1 else values
    encoded = struct.pack('<' + fmt*len(flat), *flat)
    data.extend(encoded)
    view = len(doc['bufferViews'])
    doc['bufferViews'].append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(data)-offset,
                               'target': 34963 if kind == 'SCALAR' else 34962})
    acc = {'bufferView': view, 'componentType': component_type, 'count': len(values), 'type': kind}
    if kind == 'VEC3':
        # glTF bounds describe the stored float32 data, not the preceding
        # double-precision construction (which can round outside those bounds).
        stored = list(struct.iter_unpack('<3f', encoded))
        acc.update(min=[min(p[i] for p in stored) for i in range(3)],
                   max=[max(p[i] for p in stored) for i in range(3)])
    index = len(doc['accessors'])
    doc['accessors'].append(acc)
    return index


for material, (ps, ns, uv, ids) in enumerate((body, pips)):
    doc['meshes'][0]['primitives'].append({
        'attributes': {'POSITION': accessor(ps, 3, 'VEC3'), 'NORMAL': accessor(ns, 3, 'VEC3'),
                       'TEXCOORD_0': accessor(uv, 2, 'VEC2')},
        'indices': accessor(ids, 1, 'SCALAR', 'H', 5123), 'material': material,
    })
doc['buffers'] = [{'byteLength': len(data), 'uri': 'data:application/octet-stream;base64,' + base64.b64encode(data).decode()}]
(ROOT / 'pieces/ur-dice.gltf').write_text(json.dumps(doc, separators=(',', ':')) + '\n')
catalog_path = ROOT / 'catalog.json'
catalog = json.loads(catalog_path.read_text())
entry = next(p for p in catalog['pieces'] if p['src'] == 'pieces/ur-dice.gltf')
entry['faces'] = faces
catalog_path.write_text(json.dumps(catalog, indent=2) + '\n')
# Keep explicit Play sizes in sync: stale sizes would stretch the regular mesh.
bounds = [doc['accessors'][p['attributes']['POSITION']]
          for p in doc['meshes'][0]['primitives']]
size = [(max(b['max'][i] for b in bounds)-min(b['min'][i] for b in bounds))*SCALE
        for i in range(3)]
release_path = ROOT / 'release.json'
release = json.loads(release_path.read_text())


def resize(pieces):
    for piece in pieces:
        if piece.get('src') == 'pieces/ur-dice.gltf':
            piece['size'] = size
        resize(piece.get('children', []))


resize(release['game']['children'])
release_path.write_text(json.dumps(release, indent=2) + '\n')
print('Wrote regular Ur die, catalog poses and release dimensions. Run scripts/build.py to package.')
