"""Check the generated die's pivot and all authored roll outcomes (stdlib only)."""
import base64
import json
import math
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parent.parent


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def rotate(point, angles):
    x, y, z = point
    rx, ry, rz = angles
    x, y = math.cos(rz)*x-math.sin(rz)*y, math.sin(rz)*x+math.cos(rz)*y
    x, z = math.cos(ry)*x+math.sin(ry)*z, -math.sin(ry)*x+math.cos(ry)*z
    return (x, math.cos(rx)*y-math.sin(rx)*z, math.sin(rx)*y+math.cos(rx)*z)


class DieGeometryTest(unittest.TestCase):
    def test_rolls_around_body_centre_on_four_flat_faces(self):
        entry = json.loads((ROOT / 'catalog.json').read_text())['pieces'][0]
        doc = json.loads((ROOT / entry['src']).read_text())
        data = base64.b64decode(doc['buffers'][0]['uri'].split(',')[1])

        self.assertEqual(len(data), doc['buffers'][0]['byteLength'])
        self.assertEqual(len(doc['nodes']), 1)
        self.assertNotIn('rotation', doc['nodes'][0])
        self.assertNotIn('matrix', doc['nodes'][0])

        def accessor(index):
            acc = doc['accessors'][index]
            view = doc['bufferViews'][acc['bufferView']]
            components = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[acc['type']]
            fmt = '<' + {5126: 'f', 5123: 'H'}[acc['componentType']] * components
            offset = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
            stride = view.get('byteStride', struct.calcsize(fmt))
            return [struct.unpack_from(fmt, data, offset + i*stride) for i in range(acc['count'])]

        # Accessor bounds must exactly match the encoded values.
        for i, acc in enumerate(doc['accessors']):
            view = doc['bufferViews'][acc['bufferView']]
            self.assertEqual(view.get('byteOffset', 0) % 4, 0)
            self.assertLessEqual(view.get('byteOffset', 0) + view['byteLength'], len(data))
            values = accessor(i)
            self.assertTrue(all(math.isfinite(x) for row in values for x in row))
            if 'min' in acc:
                self.assertEqual(acc['min'], [min(v[axis] for v in values) for axis in range(len(values[0]))])
                self.assertEqual(acc['max'], [max(v[axis] for v in values) for axis in range(len(values[0]))])

        node = doc['nodes'][0]
        body = doc['meshes'][node['mesh']]['primitives'][0]
        points = [tuple((p[i]*node['scale'][i]+node.get('translation', [0]*3)[i])*entry['scale'][i]
                        for i in range(3)) for p in accessor(body['attributes']['POSITION'])]
        indices = [v[0] for v in accessor(body['indices'])]
        triangles = [[points[j] for j in indices[i:i+3]] for i in range(0, len(indices), 3)]
        volumes = [dot(a, cross(b, c)) for a, b, c in triangles]
        centre = [sum(v * sum(p[i] for p in tri)/4 for v, tri in zip(volumes, triangles))/sum(volumes)
                  for i in range(3)]
        self.assertLess(math.sqrt(dot(centre, centre)), 1e-9, 'Body must rotate about its volume centroid')
        normals = []
        for a, b, c in triangles:
            n = cross([b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)])
            length = math.sqrt(dot(n, n))
            if length > 0:
                normals.append((length, [v/length for v in n]))
        main_faces = [n for area, n in normals if area > max(area for area, _ in normals)*0.1]
        self.assertEqual(len(main_faces), 4)
        self.assertEqual(sorted(f['name'] for f in entry['faces']), ['0', '0', '1', '1'])
        # Equal main triangles and tetrahedral normal angles certify regularity.
        main_triangles = []
        for a, b, c in triangles:
            n = cross([b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)])
            if math.sqrt(dot(n, n)) > max(area for area, _ in normals)*0.1:
                main_triangles.append((a, b, c))
        lengths = [math.dist(tri[i], tri[(i+1) % 3]) for tri in main_triangles for i in range(3)]
        self.assertLess(max(lengths)-min(lengths), 2e-9)
        for i, a in enumerate(main_faces):
            for b in main_faces[i+1:]:
                self.assertAlmostEqual(dot(a, b), -1/3, delta=1e-7) # float32 mesh planes
        # The bevel patches join to a closed solid, with no cracks or doubled faces.
        from collections import Counter
        edges = Counter()
        for tri in triangles:
            vertices = [tuple(round(x, 9) for x in p) for p in tri]
            for i in range(3):
                edges[tuple(sorted((vertices[i], vertices[(i+1) % 3])))] += 1
        self.assertEqual(set(edges.values()), {2})
        # UVs are regenerated and nondegenerate; normals agree with triangle winding.
        for primitive in doc['meshes'][0]['primitives']:
            ps = accessor(primitive['attributes']['POSITION'])
            ns = accessor(primitive['attributes']['NORMAL'])
            uv = accessor(primitive['attributes']['TEXCOORD_0'])
            ids = [v[0] for v in accessor(primitive['indices'])]
            self.assertEqual(len(ps), len(ns))
            self.assertEqual(len(ps), len(uv))
            self.assertTrue(all(0 <= x <= 1 for pair in uv for x in pair))
            for n in ns:
                self.assertAlmostEqual(dot(n, n), 1, places=6)
            for i in range(0, len(ids), 3):
                a, b, c = [ps[j] for j in ids[i:i+3]]
                n = cross([b[k]-a[k] for k in range(3)], [c[k]-a[k] for k in range(3)])
                self.assertGreater(dot(n, ns[ids[i]]), 0)
                a, b, c = [uv[j] for j in ids[i:i+3]]
                self.assertGreater(abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])), 1e-10)
        self.assertEqual(doc['extras']['probability']['faces'], entry['faces'])
        self.assertEqual([f['rotation'] for f in entry['faces']], [
            [-math.acos(-1/3), math.pi, 0], [-math.acos(-1/3), math.pi/3, 0],
            [-math.acos(-1/3), -math.pi/3, 0], [0, 0, 0]])
        pip_primitive = doc['meshes'][0]['primitives'][1]
        pip_points = accessor(pip_primitive['attributes']['POSITION'])
        pip_normals = accessor(pip_primitive['attributes']['NORMAL'])
        self.assertEqual(len(pip_points), 6*17)
        pip_uvs = accessor(pip_primitive['attributes']['TEXCOORD_0'])
        pip_islands = []
        for start in range(0, len(pip_points), 17):
            island_uvs = pip_uvs[start:start+17]
            pip_islands.append(([min(p[i] for p in island_uvs) for i in range(2)],
                                [max(p[i] for p in island_uvs) for i in range(2)]))
            centre_pip, normal_pip = pip_points[start], pip_normals[start]
            radii = [math.dist(centre_pip, p) for p in pip_points[start+1:start+17]]
            self.assertLess(max(radii)-min(radii), 1e-7)
            for point in pip_points[start:start+17]:
                # Positions and normals are stored as float32.
                self.assertLess(abs(dot([point[i]-centre_pip[i] for i in range(3)], normal_pip)), 1e-7)
        for i, (a_min, a_max) in enumerate(pip_islands):
            for b_min, b_max in pip_islands[i+1:]:
                self.assertTrue(any(a_max[axis] < b_min[axis] or b_max[axis] < a_min[axis]
                                    for axis in range(2)), 'Pip UV islands overlap')
        all_points = points + [tuple((p[i]*node['scale'][i]+node.get('translation', [0]*3)[i])*entry['scale'][i]
                                    for i in range(3)) for p in pip_points]
        supported_faces = []
        resting_heights = []
        for face in entry['faces']:
            # A face change must permute the regular body's vertices in place,
            # not just put some face flat. This also preserves the silhouette
            # under any common table-axis turn or camera angle.
            for point in points:
                rotated = rotate(point, face['rotation'])
                self.assertLess(min(math.dist(rotated, target) for target in points), 2e-9,
                                'Changing face changes the body outline')
            rotated_normals = [rotate(n, face['rotation']) for n in main_faces]
            support = min(range(4), key=lambda i: rotated_normals[i][1])
            self.assertAlmostEqual(rotated_normals[support][1], -1, places=10)
            supported_faces.append(support)
            resting_heights.append(-min(rotate(p, face['rotation'])[1] for p in all_points))
            upper_dots = sum(rotate(pip_points[i], face['rotation'])[1] > 0.4
                             for i in range(0, len(pip_points), 17))
            self.assertEqual(upper_dots, 3 if face['name'] == '1' else 0)
            # Ground placement changes only Y; X/Z of the body's centre must stay fixed.
            world_centre = rotate(centre, face['rotation'])
            self.assertLess(math.hypot(world_centre[0], world_centre[2]), 1e-9)
        self.assertEqual(len(set(supported_faces)), 4)
        self.assertLess(max(resting_heights)-min(resting_heights), 2e-9)
        size = [max(p[i] for p in all_points)-min(p[i] for p in all_points) for i in range(3)]
        release = json.loads((ROOT / 'release.json').read_text())
        checked_dice = []
        def check_sizes(pieces):
            for piece in pieces:
                if piece.get('src') == entry['src']:
                    checked_dice.append(piece)
                    self.assertEqual(len(piece['size']), 3)
                    self.assertIn(piece['rotation'], [face['rotation'] for face in entry['faces']])
                    for actual, expected in zip(piece['size'], size):
                        self.assertAlmostEqual(actual, expected, places=8)
                check_sizes(piece.get('children', []))
        check_sizes(release['game']['children'])
        self.assertEqual(len(checked_dice), 4)

    def test_one_point_two_mm_rounding_pip_clearance_and_continuous_face_uvs(self):
        from collections import defaultdict
        doc = json.loads((ROOT / 'pieces/ur-dice.gltf').read_text())
        data = base64.b64decode(doc['buffers'][0]['uri'].split(',')[1])
        def read(index, width):
            a = doc['accessors'][index]; v = doc['bufferViews'][a['bufferView']]
            fmt = '<' + {5126:'f',5123:'H'}[a['componentType']] * width
            return list(struct.iter_unpack(fmt, data[v['byteOffset']:v['byteOffset']+v['byteLength']]))
        body, pips = doc['meshes'][0]['primitives']
        ps = read(body['attributes']['POSITION'],3)
        ns = read(body['attributes']['NORMAL'],3)
        uv = read(body['attributes']['TEXCOORD_0'],2)
        ids = [x[0] for x in read(body['indices'],1)]
        scale = doc['nodes'][0]['scale'][0]
        radius = .0012 / scale
        # Independent exact tetrahedral support planes, in model units.
        planes = [(0,-1,0),(-math.sqrt(2/3),1/3,-math.sqrt(2)/3),
                  (0,1/3,2*math.sqrt(2)/3),(math.sqrt(2/3),1/3,-math.sqrt(2)/3)]
        seen = {}
        for p,n in zip(ps,ns):
            core = [p[k]-radius*n[k] for k in range(3)]
            gaps = [1.125/3-radius-dot(core,f) for f in planes]
            self.assertGreater(min(gaps),-1e-7)
            self.assertLess(min(abs(x) for x in gaps),1e-7)
            key=tuple(round(x,6) for x in p)
            if key in seen:self.assertLess(math.dist(n,seen[key]),1e-6)
            seen[key]=n
        slots=[int(p[0]*2)+2*int(p[1]*2) for p in uv]
        self.assertEqual(set(slots),{0,1,2,3})
        winding=defaultdict(set); coordinates={}
        for start in range(0,len(ids),3):
            tri=ids[start:start+3]
            self.assertEqual(len({slots[i] for i in tri}),1)
            a,b,c=[uv[i] for i in tri]
            area=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
            winding[slots[tri[0]]].add(area>0)
            for i in tri:
                key=(slots[i],tuple(round(x,6) for x in ps[i]))
                if key in coordinates:self.assertLess(math.dist(uv[i],coordinates[key]),1e-6)
                coordinates[key]=uv[i]
        self.assertTrue(all(len(signs)==1 for signs in winding.values()))
        for p,n in zip(read(pips['attributes']['POSITION'],3),read(pips['attributes']['NORMAL'],3)):
            # The whole circle must remain on the planar face, inside the fillet.
            core=[p[k]-(radius+.001)*n[k] for k in range(3)]
            self.assertGreater(min(1.125/3-radius-dot(core,f) for f in planes),-1e-7)

    def test_packaged_zip_matches_current_models_and_manifest(self):
        import hashlib, zipfile
        with zipfile.ZipFile(ROOT/'dist/mod.zip') as archive:
            release=json.loads(archive.read('release.json'))
            self.assertEqual(archive.read('release.json'),(ROOT/'release.json').read_bytes())
            self.assertEqual(archive.read('catalog.json'),(ROOT/'catalog.json').read_bytes())
            for item in release['files']:
                data=archive.read(item['path'])
                self.assertEqual(data,(ROOT/item['path']).read_bytes())
                self.assertEqual(len(data),item['bytes'])
                self.assertEqual(hashlib.sha256(data).hexdigest(),item['sha256'])


if __name__ == '__main__':
    unittest.main()
