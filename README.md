<p align="center"><img src="assets/icon.png" width="96" alt="TTPF Studio icon"></p>

<h1 align="center">TTPF Studio for GTA 6</h1>

<p align="center">Texture, 3D model, archive and video tool for <b>Grand Theft Auto VI (GTA 6)</b>: view, export and replace textures in <code>.ttpf</code> packs, recover files from <code>RPF9</code> archives, export 3D models to OBJ, play Bink videos, and browse everything in one searchable library.</p>

---

## What it does

**Texture packs (`.ttpf`)**
- Reads the pack index and texture table, follows each texture's mip chain and decodes the full-resolution image.
- Formats: BC1, BC2, BC3, BC4, BC5, BC6H, BC7, RGBA8, BGRA8, RGBA16F.
- Undoes the GPU swizzle (two tile layouts, picked per texture from the texture table).
- Packs with a missing or damaged header still open: textures are found by scanning, their mip levels are grouped, and format and size are guessed. A wrong guess can be corrected in the app.
- Viewer with zoom, pan, channel isolation (R / G / B / A / Split), a thumbnail grid, and a guess at what each texture is for (colour, normal map, packed data, greyscale).
- Export PNG or DDS. Replace a texture from PNG (RGBA8, BC1, BC2, BC3, BC5) or from DDS (any supported format, including BC7), then save a modded copy of the pack. The original file is never modified.

**RPF9 archives (`.rpf`)**
- The file list inside RPF9 is encrypted and this tool does not decrypt it. Much of the data inside is not encrypted, though, so the tool recovers what it can: XML and text files (rejoined from their 64 KB chunks), Bink / Bink 2 videos (cut out exactly using the size in their header), PSIN resources and other binary blobs.
- Original file names and folders are not available; recovered files are named from their content.
- Videos: **Play** opens them in RAD Video Tools; **Convert to MP4** turns them into normal H.264/AAC MP4 files. Both need [RAD Video Tools](https://www.radgametools.com/bnkdown.htm) installed (free, cannot be bundled), and converting also needs [ffmpeg](https://ffmpeg.org/). Converting needs temporary disk space for the uncompressed video, about 6 MB per 1080p frame.

**3D models**
- Finds every named model in an RPF9 archive (characters, heads, clothing, shoes, cars, boats and their tuning parts, props, map sectors, building interiors) and decodes its mesh: positions, UVs and triangles.
- Colours: each material's texture is found in the `.ttpf` packs by its hash and shown on the model; vehicle paint uses a colour you pick.
- Turntable preview (drag to turn, wheel to zoom) and **Export OBJ** for one model or every listed model, with an `.mtl` file and the textures as PNG, ready for Blender.
- Format notes on the model layout come from the [6Stars](https://github.com/Ahmet-Dedeler/6Stars) project; the decoder here is a separate implementation.

**Library**
- Indexes a whole game folder once into a catalogue (every texture of every pack, every file recovered from every archive) and sorts it into categories: peds & clothing, vehicles, weapons, props, maps, terrain, interiors, UI, effects, game data XML by type, videos, scripts, skeletons and rigged models.
- Browse by category, search everything at once, preview, open any item in the main window instantly, or export a whole selection. Re-indexing only re-reads packs that changed.

**Bulk export**
- "Export everything" walks a whole folder of packs and writes every texture as a PNG sorted into `<pack>/<Colour | Normal map | Packed data | Greyscale>/`, with an `index.csv` catalogue (format, size, file offset, content hash). It skips blank textures and duplicates, uses all CPU cores, and resumes if stopped. It can also recover RPF archive contents (`files.csv`).

## Platform support

| Source | Status |
|---|---|
| GTA 6 on Xbox Scarlett (X/S) | Assumed tested and working |
| GTA 6 on PC | Not confirmed. Textures stored without swizzle can be viewed with the *Linear* tiling option |
| GTA 6 on PS5 | Not supported if the packs use a different compressor (such as Oodle Kraken) |

## Roadmap

- **Character colours.** Ped and clothing colour maps are chosen at run time by name, and the name hash is not known yet, so characters show without colour.
- **Skeletons and animations.**
- **Vehicle wheels and map placement.** High-detail car bodies export without their wheels, and map pieces are not placed in the world yet.
- **Faster first index** for very large archives.

## Install

Needs Python 3.10+.

```powershell
pip install -r requirements.txt
python ttpf_tool.py
```

On Windows, add Start menu and Desktop shortcuts (then right-click the Start menu entry → *Pin to taskbar*):

```powershell
python ttpf_tool.py --install
```

## Command line

```powershell
python ttpf_tool.py path\to\pack.ttpf                 # open a file in the app
python ttpf_tool.py --dump <packs folder> <output>     # export every texture (add --rpf to also recover RPF contents)
python ttpf_tool.py --rpf path\to\archive.rpf          # print an RPF9 header
python ttpf_tool.py --selftest                         # run the built-in tests
```

## Safety

The app only reads your game files. Everything it writes goes to the folder you choose, to `%APPDATA%\TTPF Studio` (settings, error log) or to its install folder. Saving a modded pack always writes a new file.

## How the formats work

See [FORMATS.md](FORMATS.md) for the reverse-engineered layout of the pack index, texture table, mip chains, GPU swizzle and RPF9 archives.

## Legal

This is an unofficial fan-made tool. It is not affiliated with or endorsed by Rockstar Games or Take-Two Interactive. Grand Theft Auto, GTA and GTA VI are trademarks of Take-Two Interactive. It contains no game files, game assets or encryption keys. Use it only with files you are allowed to use.

## License

MIT, see [LICENSE](LICENSE).
