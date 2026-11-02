# Ur dice source

`generate.py` is the current source of truth. It produces the current model at
`../../pieces/ur-dice.gltf`: a regular tetrahedron with a 1.2 mm fillet, four face
UV islands including rounded rims, and six geometric white corner circles.

The imported `d4.gltf` and its maps are historical artwork references. They are
not loaded by the generator or included as the live die in the game ZIP.
Ur uses solid-colour materials; there is no printed normal map to regenerate.
For custom body artwork, add a base-colour texture and set the body material's
baseColorFactor to white so the black material tint does not darken the image.
The corner circles are separate geometry/material and may be removed or edited
independently of body painting.
