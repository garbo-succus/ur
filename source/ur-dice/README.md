# Ur dice source

`generate.py` produces `../../pieces/ur-dice.gltf`: the original regular
rounded tetrahedron with a 1.2 mm fillet, two subdivisions per bevel patch,
148 triangles, and four UV islands including their rounded rims.

`paint_texture.py` bakes the two black corner markings into an embedded 1024px
RGB PNG on a white body. The ellipsoid mask balances top and edge views.
`PAINT_RADIUS`, `PAINT_LENGTH`, and `PAINT_CENTRE` control its shape;
`EDGE_WIDTH` controls the soft antialiased paint edge. The atlas includes
8-pixel gutters and uses bilinear/trilinear filtering to avoid UV seams.

Paint adds no geometry or raised surface. The original smooth vertex normals
continue through the painted areas; a normal map is unnecessary for flat paint.
The imported `d4.gltf` and its maps remain historical references and are not
loaded by the generator or included as the live die in the game ZIP.
