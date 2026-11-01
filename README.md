*::: [🗑️💧](https://garbo.succus.games) PROUDLY PRESENTS :::*

# The Royal Game of Ur

A game for [Probability](https://probability.nz), 2600 years in the making!

![Screenshot of the Royal Game of Ur](./screenshot.png)

RELEASED UNDER [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en)

* [NEFTALY HERNANDEZ](https://neftaly.com): ART, DEV
* [JOE RICKWOOD](https://joerickwood.github.io/): DEV
* [MISHA TSYATSKO](https://misha_tsyatsko.artstation.com/): ART

## How to play

[![Tom Scott vs Irving Finkel: The Royal Game of Ur](https://img.youtube.com/vi/WZskjLq040I/maxresdefault.jpg)Tom Scott vs Irving Finkel: The Royal Game of Ur](https://youtu.be/WZskjLq040I)

## Editing and building

The current Play setup is in `release.json`. Models use metre dimensions and every piece has an explicit `size`; resting heights and stacks are resolved by Play.

Run `npm run build` to package the current working tree into `dist/mod.zip`. This includes uncommitted edits. Load that ZIP with Play → Edit → Create → Load, or open this folder through a template URL.

Regenerate the regular Ur tetrahedron with `python3 source/ur-dice/generate.py`, then validate it with `python3 scripts/test_dice.py` and package it with `npm run build`.

The generator derives the centred geometry, uniform rounded bevels, normals, UV islands and six circular corner markings from a regular tetrahedron. Its four roll poses use `-acos(-1/3)` (-109.4712206°), 180°, ±60° and zeros. It updates both `catalog.json` and the embedded glTF face metadata, plus the explicit die sizes in `release.json` so Play does not stretch the new mesh. The poses are symmetries of the entire tetrahedron: all four outcomes have the same footprint, silhouette and resting height, while the marked corner changes. Ur uses solid materials and geometric dots, so no texture images need rebaking. The earlier imported D4 files remain as historical source references; regeneration no longer reads them.
