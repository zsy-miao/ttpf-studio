# File formats

Reverse-engineered layouts. Fields marked *unknown* are kept as-is by the tool. All values are little-endian.

## `.ttpf` texture packs

### Header (0x18 bytes)

| Offset | Type | Meaning |
|---|---|---|
| 0x00 | char[4] | `RDS5` |
| 0x04 | u32 | number of index entries |
| 0x08 | u32 | virtual size (bytes) |
| 0x0C | u32 | *unknown* |
| 0x10 | u32 | *unknown* |
| 0x14 | u32 | 0 |

### Index (entry count × u64, right after the header)

| Bits | Meaning |
|---|---|
| 0–23 | file offset ÷ 256 |
| 24–29 | flags (32 = zlib stream) |
| 30–47 | virtual page number (4 KB pages) |
| 48–63 | *unknown* |

Each non-zero entry points to one zlib stream (`78 DA`). One stream is one mip level, or a packed mip tail.

### Texture table (`ETPF` block)

Found by searching for `ETPF` after the index. Layout: `ETPF`, u32 *unknown*, u32 count, u32 byte size, then `count` 16-byte records:

| Offset | Type | Meaning |
|---|---|---|
| 0x0 | u32 | bits 0–13 width − 1, bits 14–27 height − 1, bits 28–31 mip count |
| 0x4 | u32 | bits 0–9: `DXGI_FORMAT << 3` (low 3 bits *unknown*); bits 24–25: tiling (0 = standard, non-zero = 4 KB "R") |
| 0x8 | u32 | bits 0–19: first virtual page |
| 0xC | u32 | *unknown* |

`format >> 3` is a standard DXGI format number: 28 RGBA8, 71 BC1, 77 BC3, 80 BC4, 83 BC5, 87 BGRA8, 98 BC7 (sRGB variants are +1).

### Mip chains

A texture's mip levels sit one after another in virtual memory, smallest first: the stream at the first page, then the stream at `page + ceil(length / 4096)`, and so on until a stream is at least as big as mip 0. The first stream usually holds the packed mip tail. In files without a header, consecutive streams that grow by exactly 2× or 4× belong to the same texture.

### GPU swizzle

Pixels (or 4×4 blocks for BC formats) are stored in tiles. Inside a tile the element index bits interleave x and y. Tiles are laid out row-major.

| Element size | Standard micro tile | "R" micro tile |
|---|---|---|
| 4 bytes (RGBA8) | `x x y y y x` | `x x y y y x` |
| 8 bytes (BC1, BC4) | `x y x x y` | `x y y x x` |
| 16 bytes (BC2/3/5/6H/7) | `x y x y` | `y y x x` |

Read the patterns from the lowest address bit upwards. After the 256-byte micro tile, bits alternate `y x y x …` up to the tile size. Standard textures use 64 KB tiles when mip 0 is at least 64 KB, otherwise 4 KB; "R" textures use 4 KB tiles.

## RPF9 archives

| Offset | Type | Meaning |
|---|---|---|
| 0x00 | char[4] | `9FPR` (`RPF9`) |
| 0x04–0x3F | | header fields, mostly *unknown* |
| 0x40– | | encrypted file list |

The file list is encrypted, so names and folders cannot be read without keys. File data is often stored without encryption:

- **zlib streams** start on 16-byte boundaries. Files bigger than 64 KB are split into consecutive streams that each decompress to exactly 64 KB (the last one is shorter). A new file starts with a byte-order mark, `<?xml` or `PSIN`.
- **Bink videos** (`KB2?` / `BIK?`) are stored uncompressed. The u32 at +4 is the file size − 8, +8 is the frame count, +20 / +24 are width and height.
