# Index file format, version 1

Written by `HnswIndex::save`, read by `LoadedIndex::open` (`include/hnsw/index_file.h`,
`src/index_file.cpp`). The layer-0 section is the in-memory arena of `include/hnsw/layout.h`
byte for byte, so a memory-mapped file is searched in place with no translation.

## Conventions

- All integers are **little-endian**; floats are IEEE-754 binary32, little-endian.
  The loader rejects a file whose endianness marker does not read as `0x01020304`,
  and refuses to run on a big-endian host.
- `u32`/`i32`/`u64` are 4/4/8-byte unsigned/signed/unsigned integers.
- Offsets and lengths are in bytes from the start of the file unless stated otherwise.
- Every section starts on a 64-byte boundary. The writer pads with zero bytes,
  using the smallest padding that reaches the boundary, and writes no bytes after
  the last section.

## File layout

```
[0, 192)            header
[levels.offset)     node levels       count bytes
[arena.offset)      layer-0 arena     count * record_stride bytes
[upper_off.offset)  upper offsets     count * 8 bytes
[upper.offset)      upper links       4 * (number of upper words) bytes
```

Sections appear in this order, do not overlap, and each starts at or after the end of
the previous one (the first at or after byte 192). When `count = 0`, every section
has length 0 and offset 192, and the file is exactly the 192-byte header.

## Header (192 bytes)

| Offset | Size | Type | Field | Value / meaning |
|---:|---:|---|---|---|
| 0 | 8 | bytes | magic | `48 4E 53 57 49 44 58 00` (`"HNSWIDX\0"`) |
| 8 | 4 | u32 | version | `1` |
| 12 | 4 | u32 | endianness marker | `0x01020304` (on disk: `04 03 02 01`) |
| 16 | 4 | u32 | metric | `0` = squared L2 (the only metric in version 1) |
| 20 | 4 | u32 | dim | vector dimension, 1 to 65536 |
| 24 | 4 | u32 | M | max links on layers >= 1 (layer 0: 2M), 2 to 1024 |
| 28 | 4 | u32 | ef_construction | informational only; not used by the loader |
| 32 | 8 | u64 | count | number of nodes `n`, at most 2^32 - 1 |
| 40 | 4 | u32 | entry_point | node id where searches start; `0` when `n = 0` |
| 44 | 4 | i32 | max_level | top layer, 0 to 255; `-1` when `n = 0` |
| 48 | 8 | u64 | record_stride | bytes per layer-0 record; must equal `Layout::make(dim, M).record_stride` |
| 56 | 8 | u64 | total_file_size | must equal the actual file size |
| 64 | 8 | u64 | levels.offset | |
| 72 | 8 | u64 | levels.length | `n` |
| 80 | 8 | u64 | arena.offset | |
| 88 | 8 | u64 | arena.length | `n * record_stride` |
| 96 | 8 | u64 | upper_offsets.offset | |
| 104 | 8 | u64 | upper_offsets.length | `n * 8` |
| 112 | 8 | u64 | upper_links.offset | |
| 120 | 8 | u64 | upper_links.length | multiple of 4; see below |
| 128 | 60 | bytes | reserved | all zero |
| 188 | 4 | u32 | header_crc32 | CRC-32 of header bytes `[0, 188)` |

The CRC is CRC-32/ISO-HDLC (the zlib `crc32`): reflected polynomial `0xEDB88320`,
initial value and final xor `0xFFFFFFFF`. `crc32("123456789") = 0xCBF43926`.
It covers the header only; data sections are protected by deep validation, not a checksum.

## Sections

### Node levels

`n` bytes; byte `i` is the top layer of node `i` (0 to `max_level`).

### Layer-0 arena

`n` records of `record_stride` bytes; node `i` starts at `arena.offset + i * record_stride`.
With `L = Layout::make(dim, M)` (`include/hnsw/layout.h`):

| Record offset | Size | Content |
|---:|---:|---|
| 0 | 4 | u32 link count `c` (at most `2M`) |
| 4 | `8M` | `2M` u32 neighbor ids; the first `c` are valid, the rest are written as zero |
| `4 + 8M` | to `vector_offset` | zero padding |
| `vector_offset = round_up(4 + 8M, 32)` | `4 * dim` | the vector, `dim` floats |
| `vector_offset + 4 * dim` | to `record_stride` | zero padding |

`record_stride = round_up(vector_offset + 4 * dim, 64)`. Because the section and the
stride are 64-byte aligned, every vector is 32-byte aligned in memory.

Example, `dim = 128, M = 16`: count at 0, links at 4 (32 ids), vector at 160, stride 704.

### Upper offsets

`n` u64 values. For a node with level `0` the value is `0`. For a node with level
`l >= 1` it is a **word** (u32) index into the upper links section where the node's
`l` blocks start. Blocks are dense and in id order: the first node with level >= 1 has
offset 0, and each following such node starts right after the previous one's blocks.

### Upper links

A sequence of u32 words. A node with level `l >= 1` owns `l` consecutive blocks of
`1 + M` words; block `j - 1` holds layer `j` as `[count][M ids]` (count at most `M`,
unused ids zero). The section length is `4 * (1 + M) * (sum of all node levels)`.

## What the loader checks

All failures throw `hnsw::IndexFileError` (derived from `std::runtime_error`), with a
message naming the file and the problem.

**Shallow validation (always, O(1)).** The real file size from `fstat` is the ground
truth: the loader maps or allocates exactly that many bytes and never sizes anything
from an unchecked header field. In order:

1. The file is a regular file of at least 192 bytes.
2. Magic, version (`1`), endianness marker, header CRC, reserved bytes all zero, metric `0`.
3. `1 <= dim <= 65536`, `2 <= M <= 1024`, `count <= 2^32 - 1`.
4. If `count = 0`: `max_level = -1` and `entry_point = 0`. Otherwise
   `0 <= max_level <= 255` and `entry_point < count`.
5. `record_stride` equals the stride computed from `dim` and `M`.
6. `total_file_size` equals the real file size.
7. For each section in order: the offset is a multiple of 64; it is at or after the
   end of the previous section (or the header); `offset + length` does not overflow
   and is at most the file size; the length is exactly what the header implies
   (levels `n`, arena `n * record_stride`, upper offsets `8n`, upper links a multiple
   of 4). All products and sums use overflow-checked 64-bit arithmetic.

**Deep validation (`LoadOptions::validate_deep`, default on, O(file size), no allocation).**

1. Every node level is at most `max_level`; the entry point's level equals `max_level`.
2. Upper offsets: zero for level-0 nodes; otherwise exactly the dense, id-ordered
   value described above. The sum of all blocks equals the upper links section length.
3. Every link count is at most its cap (`2M` on layer 0, `M` above).
4. Every link id is `< n`, is not the node itself, and names a node whose level is at
   least the layer of the link.
5. Every vector element is finite (no NaN or infinity).

With deep validation on, searching any file that loads is memory-safe: search only
follows links, and every link, count and block offset it can reach has been checked.
Duplicate neighbors are not rejected; they are harmless to search.

**With `validate_deep = false` the file must be trusted.** Only the shallow checks run,
so a corrupt or malicious file can make search read out of bounds.

In `LoadMode::Mmap` the mapping is `PROT_READ`, `MAP_PRIVATE`. The file must not be
truncated or modified while it is loaded (validation runs once, at open). Replacing it
via `HnswIndex::save` is safe, because save renames a new file into place.

## Versioning

Any change to the bytes the writer produces, or to their meaning, requires a new
`version`. `tests/test_index_file.cpp` pins a hash of a small deterministic version-1
file to catch accidental changes.
