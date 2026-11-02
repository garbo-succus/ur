"""Bake softly antialiased corner paint into the existing UV atlas (stdlib)."""
import math
import struct
import zlib

TEXTURE_SIZE = 1024
PAINT_RADIUS = 0.125
PAINT_LENGTH = 0.20
PAINT_CENTRE = 0.85
EDGE_WIDTH = 0.008  # Model units: 0.128 mm total soft transition.
AXES = ((0, 1, 0), (math.sqrt(2/3), -1/3, math.sqrt(2)/3))


def paint_colour(point):
    squared = sum(x*x for x in point)
    distance = min(math.sqrt(max(0, squared-(height := sum(x*a for x,a in zip(point,axis)))**2)
                             + ((height-PAINT_CENTRE)*PAINT_RADIUS/PAINT_LENGTH)**2)
                   for axis in AXES)
    t = max(0, min(1, (distance-PAINT_RADIUS)/EDGE_WIDTH+0.5))
    t = t*t*(3-2*t)
    linear = 0.009+0.991*t
    # glTF base-colour images are sRGB; blend coverage in linear colour first.
    return round(255*(12.92*linear if linear<=0.0031308 else 1.055*linear**(1/2.4)-0.055))


def bake(positions, uvs, indices):
    size = TEXTURE_SIZE
    pixels = bytearray([255])*(size*size)
    covered = bytearray(size*size)
    for start in range(0,len(indices),3):
        ids = indices[start:start+3]
        a,b,c = [(uvs[i][0]*size,uvs[i][1]*size) for i in ids]
        denominator = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        ps = [positions[i] for i in ids]
        for y in range(max(0,math.floor(min(a[1],b[1],c[1]))),min(size,math.ceil(max(a[1],b[1],c[1])))):
            for x in range(max(0,math.floor(min(a[0],b[0],c[0]))),min(size,math.ceil(max(a[0],b[0],c[0])))):
                u = ((b[1]-c[1])*(x+.5-c[0])+(c[0]-b[0])*(y+.5-c[1]))/denominator
                v = ((c[1]-a[1])*(x+.5-c[0])+(a[0]-c[0])*(y+.5-c[1]))/denominator
                w = 1-u-v
                if min(u,v,w)<-1e-9:
                    continue
                point = tuple(u*ps[0][k]+v*ps[1][k]+w*ps[2][k] for k in range(3))
                offset = y*size+x
                pixels[offset] = paint_colour(point)
                covered[offset] = 1

    # Extend the island border into its gutter so bilinear/mipmap filtering
    # samples paint at UV seams, rather than the white atlas background.
    frontier = [i for i in range(size*size) if covered[i] and
                any(not covered[j] for j in neighbours(i,size))]
    for _ in range(8):
        pending = {}
        for i in frontier:
            for j in neighbours(i,size):
                if not covered[j]:
                    total,count = pending.get(j,(0,0))
                    pending[j] = (total+pixels[i],count+1)
        for i,(total,count) in pending.items():
            pixels[i] = round(total/count)
            covered[i] = 1
        frontier = list(pending)

    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))

    # glTF uses image top-left as UV origin. RGB PNG keeps broad loader support.
    rows = bytearray()
    for y in range(size):
        rows.append(0)
        for value in pixels[y*size:(y+1)*size]:
            rows.extend((value,value,value))
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',size,size,8,2,0,0,0))
            +chunk(b'IDAT',zlib.compress(rows,9))+chunk(b'IEND',b''))


def neighbours(index,size):
    x,y = index%size,index//size
    if x: yield index-1
    if x+1<size: yield index+1
    if y: yield index-size
    if y+1<size: yield index+size
