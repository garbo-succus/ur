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


def read_texture(doc):
    import zlib
    png = base64.b64decode(doc['images'][0]['uri'].split(',')[1])
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    offset = 8
    compressed = b''
    while offset<len(png):
        length = struct.unpack_from('>I',png,offset)[0]
        kind = png[offset+4:offset+8]
        chunk = png[offset+8:offset+8+length]
        if kind == b'IHDR':
            width,height,depth,colour,*_ = struct.unpack('>2I5B',chunk)
            assert (depth,colour)==(8,2)
        if kind == b'IDAT': compressed += chunk
        offset += length+12
    raw = zlib.decompress(compressed)
    stride = width*3+1
    assert len(raw)==height*stride
    assert all(raw[y*stride]==0 for y in range(height))
    return width,height,b''.join(raw[y*stride+1:(y+1)*stride] for y in range(height))


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
        primitives = doc['meshes'][node['mesh']]['primitives']
        points, triangles = [], []
        for primitive in primitives:
            local = [tuple(p[i]*node['scale'][i]*entry['scale'][i] for i in range(3))
                     for p in accessor(primitive['attributes']['POSITION'])]
            ids = [v[0] for v in accessor(primitive['indices'])]
            triangles.extend([[local[j] for j in ids[i:i+3]] for i in range(0,len(ids),3)])
            points.extend(local)
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
        # Identify the four planar supports independently of triangulation.
        support_axes = [(0,-1,0),(-math.sqrt(2/3),1/3,-math.sqrt(2)/3),
                        (0,1/3,2*math.sqrt(2)/3),(math.sqrt(2/3),1/3,-math.sqrt(2)/3)]
        main_faces = []
        for target in support_axes:
            matches = [n for area,n in normals if math.dist(n,target)<1e-6]
            self.assertTrue(matches)
            main_faces.append(target)
        self.assertEqual(len(main_faces), 4)
        self.assertEqual(sorted(f['name'] for f in entry['faces']), ['0', '0', '1', '1'])
        # Tiny contour triangles amplify float32 error in geometric normals;
        # classify planar triangles by their support plane instead.
        plane_distance = .375*node['scale'][0]*entry['scale'][0]
        face_areas = []
        for normal in main_faces:
            total = 0
            for a,b,c in triangles:
                if all(abs(dot(p,normal)-plane_distance)<1e-9 for p in (a,b,c)):
                    area = cross([b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)])
                    total += math.sqrt(dot(area,area))
            face_areas.append(total)
        self.assertLess(max(face_areas)-min(face_areas), 1e-10)
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
                self.assertGreater(abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])), 1e-14)
        self.assertEqual(doc['extras']['probability']['faces'], entry['faces'])
        self.assertEqual([f['rotation'] for f in entry['faces']], [
            [-math.acos(-1/3), math.pi, 0], [-math.acos(-1/3), math.pi/3, 0],
            [-math.acos(-1/3), -math.pi/3, 0], [0, 0, 0]])
        self.assertEqual(len(primitives), 1, 'Paint must not add geometry')
        self.assertEqual(len(triangles), 148, 'Keep the original two-step bevel mesh')
        axes = [(0,1,0), (math.sqrt(2/3),-1/3,math.sqrt(2)/3)]
        width, height, pixels = read_texture(doc)
        def sample(uv):
            x = min(width-1,max(0,int(uv[0]*width)))
            y = min(height-1,max(0,int(uv[1]*height)))
            return pixels[(y*width+x)*3]
        paint_points = [p for p,uv in zip(accessor(body['attributes']['POSITION']),
                                          accessor(body['attributes']['TEXCOORD_0'])) if sample(uv)<60]
        self.assertTrue(paint_points)
        all_points = points
        supported_faces = []
        resting_heights = []
        for face in entry['faces']:
            # Roll poses preserve the convex silhouette. Added paint-cut
            # vertices need not permute with the original rounded vertices.
            for direction in main_faces + axes + [(1,0,0),(0,0,1)]:
                self.assertAlmostEqual(max(dot(rotate(p,face['rotation']),direction) for p in points),
                                       max(dot(p,direction) for p in points), delta=2e-9)
            rotated_normals = [rotate(n, face['rotation']) for n in main_faces]
            support = min(range(4), key=lambda i: rotated_normals[i][1])
            self.assertAlmostEqual(rotated_normals[support][1], -1, places=10)
            supported_faces.append(support)
            resting_heights.append(-min(rotate(p, face['rotation'])[1] for p in all_points))
            top_painted = max(rotate(p,face['rotation'])[1] for p in paint_points) > .8
            self.assertEqual(top_painted, face['name']=='1')
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

    def test_rounding_and_continuous_normals_and_face_uvs(self):
        from collections import defaultdict
        doc = json.loads((ROOT / 'pieces/ur-dice.gltf').read_text())
        data = base64.b64decode(doc['buffers'][0]['uri'].split(',')[1])
        def read(index, width):
            a = doc['accessors'][index]; v = doc['bufferViews'][a['bufferView']]
            fmt = '<' + {5126:'f',5123:'H'}[a['componentType']] * width
            return list(struct.iter_unpack(fmt, data[v['byteOffset']:v['byteOffset']+v['byteLength']]))
        ps, ns, uv, ids = [], [], [], []
        for primitive in doc['meshes'][0]['primitives']:
            offset = len(ps)
            ps.extend(read(primitive['attributes']['POSITION'],3))
            ns.extend(read(primitive['attributes']['NORMAL'],3))
            uv.extend(read(primitive['attributes']['TEXCOORD_0'],2))
            ids.extend(x[0]+offset for x in read(primitive['indices'],1))
        scale = doc['nodes'][0]['scale'][0]
        radius = .0012 / scale
        # Independent exact tetrahedral support planes, in model units.
        planes = [(0,-1,0),(-math.sqrt(2/3),1/3,-math.sqrt(2)/3),
                  (0,1/3,2*math.sqrt(2)/3),(math.sqrt(2/3),1/3,-math.sqrt(2)/3)]
        seen = {}
        for p,n in zip(ps,ns):
            core = [p[k]-radius*n[k] for k in range(3)]
            gaps = [1.125/3-radius-dot(core,f) for f in planes]
            # Original mesh vertices retain exact analytic bevel normals.
            self.assertGreater(min(gaps), -1e-7)
            self.assertLess(min(abs(x) for x in gaps), 1e-7)
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

    def test_texture_edges_and_corner_coverage(self):
        doc = json.loads((ROOT/'pieces/ur-dice.gltf').read_text())
        width,height,pixels = read_texture(doc)
        self.assertEqual((width,height),(1024,1024))
        self.assertGreater(sum(30<v<245 for v in pixels[::3]),100,
                           'Paint needs a graded antialiased edge')
        self.assertEqual(doc['samplers'][0]['magFilter'],9729)
        self.assertEqual(doc['samplers'][0]['minFilter'],9987)
        self.assertNotIn('normalTexture',doc['materials'][0], 'Flat paint must not add a bump seam')
        data = base64.b64decode(doc['buffers'][0]['uri'].split(',')[1])
        primitive = doc['meshes'][0]['primitives'][0]
        def read(index,width):
            a = doc['accessors'][index]; v = doc['bufferViews'][a['bufferView']]
            fmt = '<'+{5126:'f',5123:'H'}[a['componentType']]*width
            return list(struct.iter_unpack(fmt,data[v['byteOffset']:v['byteOffset']+v['byteLength']]))
        ps = read(primitive['attributes']['POSITION'],3)
        uv = read(primitive['attributes']['TEXCOORD_0'],2)
        ids = [x[0] for x in read(primitive['indices'],1)]
        axes = [(0,1,0),(math.sqrt(2/3),-1/3,math.sqrt(2)/3)]
        dark = [0,0]
        seen = {}
        for start in range(0,len(ids),3):
            tri = ids[start:start+3]
            for weights in ((1,0,0),(0,1,0),(0,0,1),(1/3,1/3,1/3)):
                p = [sum(ps[i][k]*w for i,w in zip(tri,weights)) for k in range(3)]
                tex = [sum(uv[i][k]*w for i,w in zip(tri,weights)) for k in range(2)]
                x,y = min(width-1,int(tex[0]*width)), min(height-1,int(tex[1]*height))
                value = pixels[(y*width+x)*3]
                distances = []
                for axis in axes:
                    h = dot(p,axis)
                    distances.append(math.sqrt(sum((p[k]-h*axis[k])**2 for k in range(3))
                                               +((h-.85)*.125/.20)**2))
                nearest = min(range(2),key=distances.__getitem__)
                if distances[nearest]<.115:
                    self.assertLess(value,60)
                    dark[nearest] += 1
                if distances[nearest]>.135:
                    self.assertGreater(value,245)
                key = tuple(round(x,6) for x in p)
                if key in seen:
                    self.assertLess(abs(value-seen[key]),40,'Paint shows a UV seam')
                seen[key] = value
        self.assertTrue(all(count>10 for count in dark))

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
