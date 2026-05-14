//! Per-block remap trailer for `DocStoreVersion::V3`.
//!
//! A V3 block on disk is laid out as:
//!
//! ```text
//! [ compressed payload (same format as V2: docs followed by doc_pos array) ]
//! [ optional remap entries (only present when trailer_byte_len > 4)         ]
//! [ u32 trailer_byte_len  (little-endian, total bytes of this trailer       ]
//! [                       *including* the u32 itself; >= 4)                ]
//! ```
//!
//! The remap maps the field ids that appear inline in the doc records inside
//! the compressed payload to the schema field ids that the surrounding
//! index uses. The reader applies it on the fly when deserializing docs.
//!
//! `trailer_byte_len == 4`  → empty remap (encoded == target; behaves like V2)
//! `trailer_byte_len  > 4`  → remap pairs follow, format:
//!
//! ```text
//! VInt num_pairs
//! (VInt encoded_field_id, VInt target_field_id)  ×  num_pairs
//! ```
//!
//! All integers are unsigned varints (`tantivy_common::VInt`).

use std::collections::HashMap;
use std::io::{self, Read, Write};

use common::{BinarySerializable, VInt};

/// In-memory representation of a per-block field-id remap.
///
/// `lookup` returns the target field id for a given encoded id, falling back
/// to identity when no entry is present. This makes the empty remap a no-op,
/// so callers can blindly apply it during deserialization.
#[derive(Debug, Default, Clone)]
pub struct BlockFieldRemap {
    /// Sorted vector keeps lookups branch-light for tiny remaps (the typical
    /// case is fewer than a dozen entries). For now we use a HashMap; if
    /// profiling shows it matters we can switch to a sorted Vec.
    map: HashMap<u32, u32>,
}

impl BlockFieldRemap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_pairs(pairs: impl IntoIterator<Item = (u32, u32)>) -> Self {
        let map: HashMap<u32, u32> = pairs.into_iter().collect();
        BlockFieldRemap { map }
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Look up `encoded`; return identity when there's no entry.
    #[inline]
    pub fn lookup(&self, encoded: u32) -> u32 {
        self.map.get(&encoded).copied().unwrap_or(encoded)
    }

    /// True iff every entry maps to itself, i.e. the remap is functionally
    /// identity even though it carries entries. Used to skip writing trivial
    /// non-empty remaps.
    pub fn is_identity(&self) -> bool {
        self.map.iter().all(|(k, v)| k == v)
    }

    pub fn pairs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.map.iter().map(|(k, v)| (*k, *v))
    }

    /// Serialize the trailer body (i.e. the part *before* the trailing
    /// `u32 trailer_byte_len`). For an empty remap, writes nothing.
    ///
    /// Entries are emitted in sorted order by encoded id so that the
    /// resulting `.store` bytes are deterministic across runs — without
    /// this, the underlying HashMap iteration order leaks into the file
    /// and breaks reproducible builds / content-addressed segment hashing.
    pub fn serialize_body<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        let mut entries: Vec<(u32, u32)> = self.map.iter().map(|(k, v)| (*k, *v)).collect();
        entries.sort_unstable_by_key(|&(encoded, _)| encoded);
        VInt(entries.len() as u64).serialize(writer)?;
        for (encoded, target) in entries {
            VInt(encoded as u64).serialize(writer)?;
            VInt(target as u64).serialize(writer)?;
        }
        Ok(())
    }

    /// Deserialize a remap body of `body_len` bytes from `reader`.
    pub fn deserialize_body<R: Read>(reader: &mut R, body_len: usize) -> io::Result<Self> {
        if body_len == 0 {
            return Ok(BlockFieldRemap::default());
        }
        let num_pairs = VInt::deserialize(reader)?.0 as usize;
        // Each pair is two varints; even the minimum (1-byte) encoding
        // requires 2 bytes per pair, so a body of `body_len` bytes can
        // carry at most `body_len / 2` pairs (in practice far fewer
        // because the leading num_pairs varint consumes ≥ 1 byte). Cap
        // `with_capacity` to the on-disk maximum so a corrupt or hostile
        // num_pairs varint (e.g. 2^60) can't trigger an OOM allocation
        // before the loop discovers the read past EOF.
        let cap_upper_bound = body_len / 2;
        let map_cap = num_pairs.min(cap_upper_bound);
        let mut map = HashMap::with_capacity(map_cap);
        for _ in 0..num_pairs {
            let encoded = VInt::deserialize(reader)?.0 as u32;
            let target = VInt::deserialize(reader)?.0 as u32;
            map.insert(encoded, target);
        }
        Ok(BlockFieldRemap { map })
    }
}

/// Write a complete V3 block trailer (body + `u32 trailer_byte_len`) to
/// `writer`. Used by [`BlockCompressorImpl::compress_block_and_write`] when
/// the store is operating in V3 mode.
pub fn write_block_trailer<W: Write + ?Sized>(
    writer: &mut W,
    remap: &BlockFieldRemap,
) -> io::Result<()> {
    let mut body = Vec::new();
    remap.serialize_body(&mut body)?;
    // body.len() + 4 is the on-disk trailer length, encoded as a u32. A
    // remap > 4GB is implausible but a silent `as u32` truncation would
    // corrupt the trailer (reader subtracts a wrapped length from
    // block.len()); fail-loud via try_from instead.
    let trailer_byte_len = u32::try_from(body.len() + 4).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "V3 block trailer body of {} bytes exceeds u32::MAX - 4; \
                 split the remap or rebuild without stack_with_remap",
                body.len()
            ),
        )
    })?;
    writer.write_all(&body)?;
    trailer_byte_len.serialize(writer)?;
    Ok(())
}

/// Read the trailing `u32` from a V3 block to recover the trailer's byte
/// length, then parse the remap body. Returns the remap and the byte length
/// to subtract from `block.len()` to reach the compressed-payload boundary.
pub fn read_block_trailer(block: &[u8]) -> io::Result<(BlockFieldRemap, usize)> {
    if block.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "V3 block too short for trailer length",
        ));
    }
    let mut tail: &[u8] = &block[block.len() - 4..];
    let trailer_byte_len = u32::deserialize(&mut tail)? as usize;
    if trailer_byte_len < 4 || trailer_byte_len > block.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid V3 block trailer length {} (block size {})",
                trailer_byte_len,
                block.len()
            ),
        ));
    }
    let body_len = trailer_byte_len - 4;
    let body_start = block.len() - trailer_byte_len;
    let mut body: &[u8] = &block[body_start..body_start + body_len];
    let remap = BlockFieldRemap::deserialize_body(&mut body, body_len)?;
    Ok((remap, trailer_byte_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_remap_roundtrip() {
        let mut buf = Vec::new();
        let remap = BlockFieldRemap::new();
        write_block_trailer(&mut buf, &remap).unwrap();
        assert_eq!(buf.len(), 4, "empty trailer should be just the u32 length");
        // u32 LE: trailer_byte_len = 4
        assert_eq!(buf, [4u8, 0, 0, 0]);
        let (back, n) = read_block_trailer(&buf).unwrap();
        assert_eq!(n, 4);
        assert!(back.is_empty());
    }

    #[test]
    fn non_empty_remap_roundtrip() {
        let remap = BlockFieldRemap::from_pairs([(0u32, 7u32), (1, 4), (3, 3)]);
        let mut buf = Vec::new();
        write_block_trailer(&mut buf, &remap).unwrap();
        let (back, n) = read_block_trailer(&buf).unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(back.lookup(0), 7);
        assert_eq!(back.lookup(1), 4);
        assert_eq!(back.lookup(3), 3);
        // No entry → identity.
        assert_eq!(back.lookup(42), 42);
    }

    #[test]
    fn lookup_default_is_identity() {
        let remap = BlockFieldRemap::new();
        for &id in &[0u32, 1, 5, 999] {
            assert_eq!(remap.lookup(id), id);
        }
    }

    /// Regression: two `BlockFieldRemap`s built from the same set of pairs
    /// must serialize to byte-identical trailers regardless of insertion
    /// order. Without sorting in `serialize_body`, HashMap iteration order
    /// (which is randomized per process via `RandomState`) would leak into
    /// `.store` files.
    #[test]
    fn serialization_is_deterministic_across_insertion_orders() {
        let pairs_a = [(5u32, 1u32), (2, 7), (9, 0), (1, 3), (12, 4)];
        let mut pairs_b = pairs_a.to_vec();
        pairs_b.reverse();
        let remap_a = BlockFieldRemap::from_pairs(pairs_a);
        let remap_b = BlockFieldRemap::from_pairs(pairs_b);
        let mut buf_a = Vec::new();
        let mut buf_b = Vec::new();
        write_block_trailer(&mut buf_a, &remap_a).unwrap();
        write_block_trailer(&mut buf_b, &remap_b).unwrap();
        assert_eq!(
            buf_a, buf_b,
            "trailer bytes must be independent of HashMap iteration order"
        );
    }
}
