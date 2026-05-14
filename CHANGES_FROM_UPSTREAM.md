# Changes from upstream tantivy main

This branch (`virtualize-store-field-ids-on-main`) adds **cross-schema doc
store stacking** and **additive schema evolution** on top of upstream
`quickwit-oss/tantivy` main. The merge base is the `vendor tantivy main`
commit; everything below sits as 6 commits on top of it.

## TL;DR

Two new capabilities, in roughly increasing order of scope:

1. A new on-disk doc-store format (`V3`) where each compressed block can
   carry a per-block **field-id remap trailer**. The remap translates the
   encoded field ids inside the compressed payload to the surrounding
   schema's field ids at read time. This makes it possible to byte-copy
   one segment's compressed `.store` blocks into another segment under a
   different schema — no decompression, no per-doc rewrite, no
   recompression. The core re-segmentation step collapses to block-level
   I/O plus a small per-block trailer rewrite.

2. `Index::extend_schema` / `IndexWriter::extend_schema` — APIs to grow an
   open index's schema by appending new fields at the end without
   rebuilding existing segments. Old segments stay valid; they are simply
   sparse on the new fields, matching the existing "doc didn't set a
   value" semantics.

The two features compose: a re-segmenter built on `stack_with_remap` can
unify multiple indexes with different schemas into one V3 index whose
schema is the union of the inputs, then the writer can keep extending
that schema as new sources arrive.

## Backwards compatibility

| Reader version | What it can open                |
| -------------- | ------------------------------- |
| V1             | V1 only                         |
| V2             | V1, V2                          |
| V3 *(new)*     | V1, V2, V3                      |

| Writer output | What it can stack                                    |
| ------------- | ---------------------------------------------------- |
| V2            | V2 (byte-copy), V3 (per-block trailer-strip)         |
| V3 *(default)*| V2 (per-block trailer-injection), V3 (byte-copy)     |

`DOC_STORE_VERSION` (the constant new writers stamp into footers) is
`V3` post-patch. Fresh writes go to V3 with empty trailers — byte-equivalent
to V2 except for an extra 4-byte trailer per block. V1 sources are
rejected by the stacking paths because V1 stored datetime values as i64
microseconds while V2/V3 use nanoseconds; a byte-copy would silently
rescale every datetime by 1000x.

## Public API additions

### `tantivy::store::BlockFieldRemap`

```rust
pub struct BlockFieldRemap { /* opaque */ }

impl BlockFieldRemap {
    pub fn new() -> Self;
    pub fn from_pairs(pairs: impl IntoIterator<Item = (u32, u32)>) -> Self;
    pub fn lookup(&self, encoded: u32) -> u32; // identity fallback
    pub fn is_empty(&self) -> bool;
    pub fn is_identity(&self) -> bool;
    pub fn len(&self) -> usize;
    pub fn pairs(&self) -> impl Iterator<Item = (u32, u32)> + '_;
}
```

In-memory representation of a per-block remap. `lookup` returns the target
field id for a given encoded id, falling back to identity when no entry is
present — so the empty remap is a no-op at read time and freshly-written
V3 blocks decode byte-equivalently to V2.

### `tantivy::store::StoreWriter::stack_with_remap`

```rust
impl StoreWriter {
    pub fn stack_with_remap(
        &mut self,
        store_reader: StoreReader,
        remap: BlockFieldRemap,
    ) -> io::Result<()>;
}
```

Stacks every block of `store_reader` into this writer with a per-block
trailer carrying the (composed) remap. The compressed payload of each
source block is byte-copied to the target store; only the trailer is
rewritten.

`remap` is keyed by the **source schema's logical field ids** even when
the source itself was produced by an earlier translating stack. The
caller always thinks in `source_schema → target_schema` terms;
composition with any prior per-block trailer is done internally.

Constraints:

- `self.output_version` must be ≥ V3 (the trailer is the V3 mechanism).
- Source must be V2 or V3 (V1 is rejected — see the datetime hazard
  above).
- Each `(encoded, target)` pair in `remap` must point to an in-range,
  *stored* field in the target schema. The serializer would otherwise
  drop values silently or panic on an out-of-bounds index.

### `tantivy::Index::extend_schema`

```rust
impl Index {
    pub fn extend_schema(&mut self, new_schema: Schema) -> crate::Result<()>;
}
```

Atomically grows the index's schema by appending fields at the end.
Validates that `new_schema` is a **strict prefix extension** of the
current schema:

- ≥ the current number of fields
- Each existing field at the same position has the same name, type, and
  options as before
- Reordering, removing, or modifying an existing field returns a
  `SchemaError` and leaves `meta.json` untouched

Existing segments stay valid: their stored docs encode the old field ids,
which remain correctly positioned under the extended schema. The new
fields are simply sparse on those segments.

Acquires the `INDEX_WRITER_LOCK` for the duration of the read-modify-write
on `meta.json`. Use the writer-aware sibling below if you already hold an
open `IndexWriter`.

### `tantivy::IndexWriter::extend_schema`

```rust
impl<D> IndexWriter<D> {
    pub fn extend_schema(
        &mut self,
        new_schema: Schema,
    ) -> crate::Result<Schema>;
}
```

Mid-batch schema extension on a running writer:

1. Flushes pending docs as a final batch under the OLD schema (those
   segments are sparse on whatever new fields are about to be added).
2. Drains any merges that the flush's `consider_merge_options` scheduled,
   so their `end_merge → save_metas` can't race with the schema rewrite.
3. Validates + atomically persists the extended schema via the no-lock
   internal counterpart (the writer already holds `INDEX_WRITER_LOCK`).
4. Propagates the new schema to the `SegmentUpdater` so subsequent
   commits and merges see it.
5. Drains and respawns indexing workers so future `add_document` calls
   accept the new fields.

Returns the extended `Schema` for the caller's convenience — `Index` and
`IndexWriter` carry independent schema clones, so without the returned
value the caller can't see the new fields through the outer `Index`
handle until reopen.

## V3 block layout

```
┌──────────────────────────────────────────────────────────┐
│ compressed payload (same wire format as V2)              │
│                                                          │
│   VInt num_fields                                        │
│   ┌──────────────────────────────────────────┐           │
│   │  u32 field_id (gets remapped on read)    │           │
│   │  ValueType byte                          │   × N    │
│   │  value payload (type-specific)           │           │
│   └──────────────────────────────────────────┘           │
│   …                                                      │
│   u32 doc_pos_index                                      │
├──────────────────────────────────────────────────────────┤
│ optional remap entries (only when trailer_byte_len > 4)  │
│                                                          │
│   VInt num_pairs                                         │
│   ┌──────────────────────────────────────┐               │
│   │ VInt encoded_field_id                │   × K        │
│   │ VInt target_field_id                 │               │
│   └──────────────────────────────────────┘               │
│                                                          │
│   Entries are emitted sorted by encoded_field_id so the  │
│   serialized form is byte-deterministic across runs      │
│   (HashMap iteration order does NOT leak to disk).       │
├──────────────────────────────────────────────────────────┤
│ u32 trailer_byte_len (little-endian, includes itself)    │
│                                                          │
│   == 4   → empty remap, read like V2                     │
│   >  4   → remap_body of (trailer_byte_len - 4) bytes    │
│            precedes this u32                             │
└──────────────────────────────────────────────────────────┘
```

The store footer's existing `doc_store_version` field is used as the
dispatch: a V3 reader strips the trailer (if any), then decompresses
exactly like V2; a V1/V2 reader sees no trailer at all (those store
formats are unchanged).

## How reads apply the remap

`BinaryDocumentDeserializer::from_reader_with_remap` accepts an optional
`&BlockFieldRemap`. Inside `next_field`, the encoded `field_id` read from
the compressed payload is rewritten via `remap.lookup(...)` before being
returned to the caller. Empty or `None` remap is a no-op fast path.

`StoreReader::get` and `StoreReader::iter` pair every block with its
trailer (cached together in the LRU; see *LRU and Arc-wrap* below) and
pass the remap through. The merger uses a different path —
`iter_doc_bytes_translated`, described next.

## How the merger handles remapped segments

The upstream merger's `write_storable_fields` has two paths:

- **Fast stack** — byte-copies the entire compressed `.store` of a
  source segment into the merged segment (cheap, no decompression).
- **Per-doc byte copy** — reads each doc's raw bytes via `iter_raw` and
  appends them via `store_bytes`. Falls back here on deletes,
  fewer-than-six blocks, or compressor mismatch.

A segment produced by `stack_with_remap` has source-schema field ids in
its compressed bytes and relies on its per-block trailer to translate
them at read time. The per-doc byte-copy path would drop that trailer
when it appends bytes into a fresh empty-remap target block — silently
corrupting every field id in the merged segment.

Instead, the merger now detects this case via
`StoreReader::has_non_identity_remap()` and switches to
`iter_doc_bytes_translated`:

- For blocks whose remap is empty or identity, returns the raw doc bytes
  unchanged (zero-cost fast path).
- For blocks with a real translation, decodes each doc through the remap,
  validates that every (target, encoded-after-remap) field id is in
  range and stored in the target schema, then re-encodes against the
  target schema. The resulting bytes go into a fresh empty-remap target
  block.

`has_non_identity_remap` is implemented as a tiny FileSlice read of the
last 4 bytes of each block (the `u32 trailer_byte_len`) plus, when the
trailer is non-empty, a body parse to check `is_identity()`. No full
block fetches happen during the precheck — critical for remote/network
directories where every fetch is an HTTP call.

## How extend_schema propagates through the writer

`IndexWriter` and the `SegmentUpdater` each hold their own `Index` clone
with a frozen `schema: Schema` field. `Index::extend_schema` mutates the
writer's clone but the segment updater's clone is invisible behind
`Arc<InnerSegmentUpdater>`. Without explicit propagation, the next
`save_metas` (whether from a commit or an `end_merge`) reads
`segment_updater.index.schema()` — the stale clone — and writes the OLD
schema back to `meta.json`, silently reverting the extension on the next
process restart.

The fix is a separate `RwLock<Schema>` (`current_schema`) on
`InnerSegmentUpdater`, initialized to `index.schema()` at create time
and overridden by `IndexWriter::extend_schema` via
`SegmentUpdater::set_schema`. Both `save_metas` and the `merge` helper
read from `current_schema` instead of `index.schema()`. The `merge`
helper also clones the `Index` and rewrites its in-memory schema before
calling `IndexMerger::open`, so newly-created merged segments use the
extended schema for postings, fast fields, and fieldnorms.

Merging segments whose schemas differ in field count requires the
merger to handle "field in target schema but not in this source
segment". `write_fieldnorms` now treats a missing fieldnorm column as a
constant-zero reader — the same semantics that already apply to docs
which never set a value for the field. (Fast-field and postings merging
already handle missing columns gracefully via the existing columnar
machinery.)

## LRU cache and Arc-wrap

The store's LRU caches `(Block, Arc<BlockFieldRemap>)` pairs instead of
plain `Block`. Three consequences:

1. **Cache hit returns the remap with zero I/O.** Previously the remap
   was re-parsed from the FileSlice on every hit, doubling read traffic
   on remote/network directories.

2. **`iter_raw_with_remap` clones an `Arc` per doc, not a `HashMap`.**
   The previous design's `BlockFieldRemap` clone was O(remap_size) per
   yielded doc; an `Arc` clone is O(1).

3. **An identity-remap fast path runs alongside the empty-remap one.**
   `iter_doc_bytes_translated` and `has_non_identity_remap` both use
   `BlockFieldRemap::is_identity()` to skip the decode/re-encode loop
   when every `(encoded, target)` pair maps `k → k`.

## Quickwit / async path

`read_block_async` / `read_block_with_remap_async` mirror the sync path
exactly: the V3 trailer is stripped before bytes reach the decompressor,
and `get_async` threads the remap through
`BinaryDocumentDeserializer::from_reader_with_remap`. Without this, a V3
source segment in a Quickwit-style remote-FS deployment would fail
decompression (trailer bytes fed to LZ4/zstd) and even on success would
return docs with source-schema field ids.

## Determinism

`BlockFieldRemap` is internally a `HashMap`, but its serialization is
sorted by `encoded_field_id`. Two `BlockFieldRemap`s built from the same
set of pairs (in any insertion order) produce byte-identical trailers.
This matters for reproducible builds and content-addressed segment
hashing — without sorting, `RandomState`-randomized HashMap iteration
leaks into the file.

`write_block_trailer` uses `u32::try_from` for the trailer-length write
(instead of `as u32`), so a pathological >4GB remap fails loudly rather
than wrapping to a corrupt length.

`BlockFieldRemap::deserialize_body` bounds the `HashMap::with_capacity`
allocation by `body_len / 2`, since each pair needs at least two bytes
on disk. A corrupt or hostile `num_pairs` varint can't trigger an OOM
allocation before the loop discovers the read past EOF.

## Caveats and limitations

- **`extend_schema` is additive-prefix only.** Reordering, removing, or
  modifying existing fields is rejected. Use a doc-by-doc rebuild for
  those changes.

- **`stack_with_remap` does NOT translate non-store components.** The
  doc-store byte-copy is one of several components per segment; the
  inverted index, fast fields, fieldnorms, and other columnar data each
  encode field ids inline and would need parallel translation to be
  merged across schemas. In this fork, that work lives in the
  re-segmenter crate (out of scope for the doc-store patches).

- **V1 sources are rejected by both stack paths.** Migrate a V1 segment
  via a doc-by-doc rebuild before stacking.

- **`stack_with_remap` requires a V3+ target.** A V2 writer can't carry
  the trailer that holds the translation.

- **`is_stored` mismatch is rejected.** A remap entry targeting a
  non-stored field would cause the serializer to silently drop the
  value. The caller (typically the re-segmenter) must map only between
  stored fields.

## Commits

```
0646171 Second round of fixes from independent review
383ffc6 Fixes from independent review
03859cb Add IndexWriter::extend_schema for mid-batch schema extension
62c7d33 V2→V3 cross-version stack: legacy indexes mergeable into new ones
ddf40c6 Add Index::extend_schema for additive schema evolution
2ce1928 Add DocStoreVersion::V3 with per-block field-id remap trailer
```

The two "fixes from independent review" commits fold in changes from
seven separate code reviews (four reviewers in each round). Topics
addressed: merger handling of remapped segments, schema propagation to
the segment updater, the quickwit async path, LRU cache behavior, V3
trailer determinism, V1 rejection, the writer lock around
`extend_schema`, the bounds check in `iter_doc_bytes_translated`, the
commit/extend_schema race, the field-norms gap in `extend_schema +
merge`, the `output_version` dispatch in `stack`, the `has_non_identity_remap`
perf, raw-bytes API hazards, the trailer-length cast, and the
`is_stored` validation. Before squashing for an upstream PR these can
be folded into the four feature commits.

## Test coverage

- 1087 lib tests pass on the default feature set.
- 1082 lib tests pass with the `quickwit` feature.
- New regression tests cover: V3 round-trip, V2→V3 cross-version stack,
  V3→V2 trailer-strip, V1 rejection on both stack paths,
  `stack_with_remap` rejecting V2 output, the chained-stack composition
  contract, `iter_doc_bytes_translated` respecting alive_bitset, the
  out-of-range remap target bounds check, deterministic trailer
  serialization, `IndexWriter::extend_schema` schema persistence after
  reopen, and `extend_schema` survival across commit + merge cycles.

The companion re-segmenter crate
(`crates/resegmenter/` in the outer workspace) exercises the full
cross-schema merging pipeline on top of these patches and has its own
8-test integration suite covering positional remap, type-collision
splits, asymmetric source/target schemas, tombstones, fast-field
queryability after re-segmenting, multi-source consolidation, an added
field, and a large-index fast-copy path.
