use std::fmt::Display;
use std::io;
use std::iter::Sum;
use std::num::NonZeroUsize;
use std::ops::{AddAssign, Range};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use common::{BinarySerializable, OwnedBytes};
use lru::LruCache;

use super::footer::DocStoreFooter;
use super::index::SkipIndex;
use super::Decompressor;
use crate::directory::FileSlice;
use crate::error::DataCorruption;
use crate::fastfield::AliveBitSet;
use crate::schema::document::{
    BinaryDocumentDeserializer, BinaryDocumentSerializer, DocumentDeserialize,
};
use crate::space_usage::StoreSpaceUsage;
use crate::store::index::Checkpoint;
use crate::DocId;
#[cfg(feature = "quickwit")]
use crate::Executor;

pub(crate) const DOCSTORE_CACHE_CAPACITY: usize = 100;

type Block = OwnedBytes;

/// The format version of the document store.
///
/// **V3** adds a per-block remap trailer: `[compressed_payload]
/// [optional VInt-encoded (encoded_field, target_field) remap pairs]
/// [u32 trailer_byte_len]`. When `trailer_byte_len == 4` the remap is empty
/// and reads behave like V2 (encoded field ids ARE the schema field ids).
/// When non-empty, every field id encountered inside the doc record is
/// translated through the remap on read. This allows `StoreWriter` to stack
/// another segment's compressed `.store` blocks byte-for-byte under a
/// different target schema — the heavy lift for re-segmenting across
/// incompatible schemas. See `tantivy/src/store/block_trailer.rs`.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(crate) enum DocStoreVersion {
    V1 = 1,
    V2 = 2,
    V3 = 3,
}
impl Display for DocStoreVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocStoreVersion::V1 => write!(f, "V1"),
            DocStoreVersion::V2 => write!(f, "V2"),
            DocStoreVersion::V3 => write!(f, "V3"),
        }
    }
}
impl BinarySerializable for DocStoreVersion {
    fn serialize<W: io::Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        (*self as u32).serialize(writer)
    }

    fn deserialize<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        Ok(match u32::deserialize(reader)? {
            1 => DocStoreVersion::V1,
            2 => DocStoreVersion::V2,
            3 => DocStoreVersion::V3,
            v => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid doc store version {v}"),
                ))
            }
        })
    }
}

/// Reads document off tantivy's [`Store`](./index.html)
pub struct StoreReader {
    decompressor: Decompressor,
    doc_store_version: DocStoreVersion,
    data: FileSlice,
    skip_index: Arc<SkipIndex>,
    space_usage: StoreSpaceUsage,
    cache: BlockCache,
}

/// The cache for decompressed blocks, paired with their per-block
/// field-id remap. Identity remap (the common V1/V2/fresh-V3 case) is
/// represented by the shared default `Arc<BlockFieldRemap>` so the
/// payload of a cache entry stays Block-sized except when a translating
/// stack actually populated the trailer.
type CachedBlock = (Block, Arc<super::block_trailer::BlockFieldRemap>);
struct BlockCache {
    cache: Option<Mutex<LruCache<usize, CachedBlock>>>,
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
}

impl BlockCache {
    fn get_from_cache(&self, pos: usize) -> Option<CachedBlock> {
        if let Some(entry) = self
            .cache
            .as_ref()
            .and_then(|cache| cache.lock().unwrap().get(&pos).cloned())
        {
            self.cache_hits.fetch_add(1, Ordering::SeqCst);
            return Some(entry);
        }
        self.cache_misses.fetch_add(1, Ordering::SeqCst);
        None
    }

    fn put_into_cache(&self, pos: usize, block: Block, remap: Arc<super::block_trailer::BlockFieldRemap>) {
        if let Some(cache) = self.cache.as_ref() {
            cache.lock().unwrap().put(pos, (block, remap));
        }
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            num_entries: self.len(),
        }
    }

    fn len(&self) -> usize {
        self.cache
            .as_ref()
            .map_or(0, |cache| cache.lock().unwrap().len())
    }

    #[cfg(test)]
    fn peek_lru(&self) -> Option<usize> {
        self.cache
            .as_ref()
            .and_then(|cache| cache.lock().unwrap().peek_lru().map(|(&k, _)| k))
    }
}

#[derive(Debug, Default)]
/// CacheStats for the `StoreReader`.
pub struct CacheStats {
    /// The number of entries in the cache
    pub num_entries: usize,
    /// The number of cache hits.
    pub cache_hits: usize,
    /// The number of cache misses.
    pub cache_misses: usize,
}

impl AddAssign for CacheStats {
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            num_entries: self.num_entries + other.num_entries,
            cache_hits: self.cache_hits + other.cache_hits,
            cache_misses: self.cache_misses + other.cache_misses,
        };
    }
}

impl Sum for CacheStats {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut first = iter.next().unwrap_or_default();
        for el in iter {
            first += el;
        }
        first
    }
}

impl StoreReader {
    /// Opens a store reader
    ///
    /// `cache_num_blocks` sets the number of decompressed blocks to be cached in an LRU.
    /// The size of blocks is configurable, this should be reflexted in the
    pub fn open(store_file: FileSlice, cache_num_blocks: usize) -> io::Result<StoreReader> {
        let (footer, data_and_offset) = DocStoreFooter::extract_footer(store_file)?;

        let (data_file, offset_index_file) = data_and_offset.split(footer.offset as usize);
        let index_data = offset_index_file.read_bytes()?;
        let space_usage =
            StoreSpaceUsage::new(data_file.num_bytes(), offset_index_file.num_bytes());
        let skip_index = SkipIndex::open(index_data);
        Ok(StoreReader {
            decompressor: footer.decompressor,
            doc_store_version: footer.doc_store_version,
            data: data_file,
            cache: BlockCache {
                cache: NonZeroUsize::new(cache_num_blocks)
                    .map(|cache_num_blocks| Mutex::new(LruCache::new(cache_num_blocks))),
                cache_hits: Default::default(),
                cache_misses: Default::default(),
            },
            skip_index: Arc::new(skip_index),
            space_usage,
        })
    }

    pub(crate) fn block_checkpoints(&self) -> impl Iterator<Item = Checkpoint> + '_ {
        self.skip_index.checkpoints()
    }

    pub(crate) fn decompressor(&self) -> Decompressor {
        self.decompressor
    }

    /// Returns the cache hit and miss statistics of the store reader.
    pub(crate) fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Get checkpoint for `DocId`. The checkpoint can be used to load a block containing the
    /// document.
    ///
    /// Advanced API. In most cases use [`get`](Self::get).
    fn block_checkpoint(&self, doc_id: DocId) -> crate::Result<Checkpoint> {
        self.skip_index.seek(doc_id).ok_or_else(|| {
            crate::TantivyError::InvalidArgument(format!("Failed to lookup Doc #{doc_id}."))
        })
    }

    pub(crate) fn block_data(&self) -> io::Result<OwnedBytes> {
        self.data.read_bytes()
    }

    /// On-disk format version of this store. Mainly useful to a
    /// translating-`stack` path that needs to know whether source blocks
    /// already carry V3 trailers.
    pub(crate) fn doc_store_version(&self) -> DocStoreVersion {
        self.doc_store_version
    }

    fn get_compressed_block(&self, checkpoint: &Checkpoint) -> io::Result<OwnedBytes> {
        self.data.slice(checkpoint.byte_range.clone()).read_bytes()
    }

    /// Split a V3 block-bytes slice into `(compressed_payload, remap)`.
    ///
    /// For V1/V2 (no trailer) we treat the whole slice as compressed payload
    /// and return an empty remap (= identity, no-op at field-id lookup time).
    fn split_block_trailer(
        &self,
        block_bytes: OwnedBytes,
    ) -> io::Result<(OwnedBytes, super::block_trailer::BlockFieldRemap)> {
        if self.doc_store_version < DocStoreVersion::V3 {
            return Ok((block_bytes, super::block_trailer::BlockFieldRemap::default()));
        }
        let (remap, trailer_byte_len) =
            super::block_trailer::read_block_trailer(block_bytes.as_ref())?;
        let payload_end = block_bytes.len() - trailer_byte_len;
        Ok((block_bytes.slice(0..payload_end), remap))
    }

    /// Loads and decompresses a block.
    ///
    /// Advanced API. In most cases use [`get`](Self::get).
    fn read_block(&self, checkpoint: &Checkpoint) -> io::Result<Block> {
        let (block, _remap) = self.read_block_with_remap(checkpoint)?;
        Ok(block)
    }

    /// Load the block and also return the per-block field remap. The remap
    /// is identity (no-op) on V1/V2 blocks and on V3 blocks written by a
    /// normal `StoreWriter`; only blocks emitted by a translating-`stack`
    /// path carry non-identity entries.
    ///
    /// On a cache hit the FileSlice is not touched at all — both the
    /// decompressed payload and the per-block remap are read from the LRU
    /// entry. This keeps remote/network directories (Quickwit-style FS)
    /// from paying for a fresh range fetch on every `get`.
    fn read_block_with_remap(
        &self,
        checkpoint: &Checkpoint,
    ) -> io::Result<(Block, Arc<super::block_trailer::BlockFieldRemap>)> {
        let cache_key = checkpoint.byte_range.start;
        if let Some((cached_block, cached_remap)) = self.cache.get_from_cache(cache_key) {
            return Ok((cached_block, cached_remap));
        }

        let block_bytes = self.get_compressed_block(checkpoint)?;
        let (compressed_payload, remap) = self.split_block_trailer(block_bytes)?;
        let decompressed_block =
            OwnedBytes::new(self.decompressor.decompress(compressed_payload.as_ref())?);
        let remap_arc = Arc::new(remap);

        self.cache
            .put_into_cache(cache_key, decompressed_block.clone(), Arc::clone(&remap_arc));

        Ok((decompressed_block, remap_arc))
    }

    /// Reads a given document.
    ///
    /// Calling `.get(doc)` is relatively costly as it requires
    /// decompressing a compressed block. The store utilizes a LRU cache,
    /// so accessing docs from the same compressed block should be faster.
    /// For that reason a store reader should be kept and reused.
    ///
    /// It should not be called to score documents
    /// for instance.
    pub fn get<D: DocumentDeserialize>(&self, doc_id: DocId) -> crate::Result<D> {
        let checkpoint = self.block_checkpoint(doc_id)?;
        let (block, remap) = self.read_block_with_remap(&checkpoint)?;
        let mut doc_bytes = Self::get_document_bytes_from_block(block, doc_id, &checkpoint)?;
        let remap_ref = if remap.is_empty() { None } else { Some(remap.as_ref()) };
        let deserializer = BinaryDocumentDeserializer::from_reader_with_remap(
            &mut doc_bytes,
            self.doc_store_version,
            remap_ref,
        )
        .map_err(crate::TantivyError::from)?;
        D::deserialize(deserializer).map_err(crate::TantivyError::from)
    }

    /// Returns raw bytes of a given document.
    ///
    /// Calling `.get(doc)` is relatively costly as it requires
    /// decompressing a compressed block. The store utilizes a LRU cache,
    /// so accessing docs from the same compressed block should be faster.
    /// For that reason a store reader should be kept and reused.
    pub fn get_document_bytes(&self, doc_id: DocId) -> crate::Result<OwnedBytes> {
        let checkpoint = self.block_checkpoint(doc_id)?;
        let block = self.read_block(&checkpoint)?;
        Self::get_document_bytes_from_block(block, doc_id, &checkpoint)
    }

    /// Advanced API.
    ///
    /// In most cases use [`get_document_bytes`](Self::get_document_bytes).
    fn get_document_bytes_from_block(
        block: OwnedBytes,
        doc_id: DocId,
        checkpoint: &Checkpoint,
    ) -> crate::Result<OwnedBytes> {
        let doc_pos = doc_id - checkpoint.doc_range.start;

        let range = block_read_index(&block, doc_pos)?;
        Ok(block.slice(range))
    }

    /// Iterator over all Documents in their order as they are stored in the doc store.
    /// Use this, if you want to extract all Documents from the doc store.
    /// The `alive_bitset` has to be forwarded from the `SegmentReader` or the results may be wrong.
    pub fn iter<'a: 'b, 'b, D: DocumentDeserialize>(
        &'b self,
        alive_bitset: Option<&'a AliveBitSet>,
    ) -> impl Iterator<Item = crate::Result<D>> + 'b {
        self.iter_raw_with_remap(alive_bitset)
            .map(|res| {
                let (mut doc_bytes, remap) = res?;
                let remap_ref = if remap.is_empty() { None } else { Some(remap.as_ref()) };
                let deserializer = BinaryDocumentDeserializer::from_reader_with_remap(
                    &mut doc_bytes,
                    self.doc_store_version,
                    remap_ref,
                )
                .map_err(crate::TantivyError::from)?;
                D::deserialize(deserializer).map_err(crate::TantivyError::from)
            })
    }

    /// Internal: iterate raw doc bytes paired with their block's remap.
    /// Used to feed [`iter`] so each doc record's field ids are translated
    /// against the right per-block table.
    fn iter_raw_with_remap<'a: 'b, 'b>(
        &'b self,
        alive_bitset: Option<&'a AliveBitSet>,
    ) -> impl Iterator<Item = crate::Result<(OwnedBytes, Arc<super::block_trailer::BlockFieldRemap>)>> + 'b
    {
        let last_doc_id = self
            .block_checkpoints()
            .last()
            .map(|checkpoint| checkpoint.doc_range.end)
            .unwrap_or(0);
        let mut checkpoint_block_iter = self.block_checkpoints();
        let mut curr_checkpoint = checkpoint_block_iter.next();
        // load_block returns (block_bytes, remap) for the current checkpoint.
        let load_block = |cp: Option<&Checkpoint>| {
            cp.map(|checkpoint| {
                self.read_block_with_remap(checkpoint)
                    .map_err(|e| e.kind())
            })
        };
        let mut curr_block = load_block(curr_checkpoint.as_ref());
        let mut doc_pos = 0u32;
        (0..last_doc_id).filter_map(move |doc_id| {
            if doc_id >= curr_checkpoint.as_ref().unwrap().doc_range.end {
                curr_checkpoint = checkpoint_block_iter.next();
                curr_block = load_block(curr_checkpoint.as_ref());
                doc_pos = 0;
            }
            let alive = alive_bitset
                .map(|bitset| bitset.is_alive(doc_id))
                .unwrap_or(true);
            let res = if alive {
                Some((curr_block.clone(), doc_pos))
            } else {
                None
            };
            doc_pos += 1;
            res
        })
        .map(move |(block, doc_pos)| {
            let (block_bytes, remap) = block
                .ok_or_else(|| {
                    DataCorruption::comment_only(
                        "the current checkpoint in the doc store iterator is none, this \
                         should never happen",
                    )
                })?
                .map_err(|error_kind| {
                    std::io::Error::new(error_kind, "error when reading block in doc store")
                })?;
            let range = block_read_index(&block_bytes, doc_pos)?;
            Ok((block_bytes.slice(range), remap))
        })
    }

    /// Iterator over all raw Documents in their order as they are stored in the doc store.
    /// Use this, if you want to extract all Documents from the doc store.
    /// The `alive_bitset` has to be forwarded from the `SegmentReader` or the results may be wrong.
    pub(crate) fn iter_raw<'a: 'b, 'b>(
        &'b self,
        alive_bitset: Option<&'a AliveBitSet>,
    ) -> impl Iterator<Item = crate::Result<OwnedBytes>> + 'b {
        let last_doc_id = self
            .block_checkpoints()
            .last()
            .map(|checkpoint| checkpoint.doc_range.end)
            .unwrap_or(0);
        let mut checkpoint_block_iter = self.block_checkpoints();
        let mut curr_checkpoint = checkpoint_block_iter.next();
        let mut curr_block = curr_checkpoint
            .as_ref()
            .map(|checkpoint| self.read_block(checkpoint).map_err(|e| e.kind())); // map error in order to enable cloning
        let mut doc_pos = 0;
        (0..last_doc_id)
            .filter_map(move |doc_id| {
                // filter_map is only used to resolve lifetime issues between the two closures on
                // the outer variables

                // check move to next checkpoint
                if doc_id >= curr_checkpoint.as_ref().unwrap().doc_range.end {
                    curr_checkpoint = checkpoint_block_iter.next();
                    curr_block = curr_checkpoint
                        .as_ref()
                        .map(|checkpoint| self.read_block(checkpoint).map_err(|e| e.kind()));
                    doc_pos = 0;
                }

                let alive = alive_bitset
                    .map(|bitset| bitset.is_alive(doc_id))
                    .unwrap_or(true);
                let res = if alive {
                    Some((curr_block.clone(), doc_pos))
                } else {
                    None
                };
                doc_pos += 1;
                res
            })
            .map(move |(block, doc_pos)| {
                let block = block
                    .ok_or_else(|| {
                        DataCorruption::comment_only(
                            "the current checkpoint in the doc store iterator is none, this \
                             should never happen",
                        )
                    })?
                    .map_err(|error_kind| {
                        std::io::Error::new(error_kind, "error when reading block in doc store")
                    })?;

                let range = block_read_index(&block, doc_pos)?;
                Ok(block.slice(range))
            })
    }

    /// Like [`iter_raw`](Self::iter_raw) but rewrites per-block field-id
    /// remap trailers into the doc bytes themselves, so the returned bytes
    /// encode the surrounding (target) schema's field ids directly. Used by
    /// the merger when copying docs from a segment produced by a translating
    /// `stack_with_remap` into a fresh empty-remap block — without this
    /// translation, the bytes would carry the source schema's encoded field
    /// ids and the per-block trailer would be silently dropped, corrupting
    /// the merged segment.
    ///
    /// Identity remap blocks (the common case) short-circuit to a raw byte
    /// copy with no decode/re-encode overhead.
    pub(crate) fn iter_doc_bytes_translated<'a: 'b, 'b>(
        &'b self,
        target_schema: &'b crate::schema::Schema,
        alive_bitset: Option<&'a AliveBitSet>,
    ) -> impl Iterator<Item = crate::Result<Vec<u8>>> + 'b {
        let version = self.doc_store_version;
        self.iter_raw_with_remap(alive_bitset).map(move |res| {
            let (mut doc_bytes, remap) = res?;
            if remap.is_empty() {
                return Ok(doc_bytes.as_slice().to_vec());
            }
            let deserializer = BinaryDocumentDeserializer::from_reader_with_remap(
                &mut doc_bytes,
                version,
                Some(remap.as_ref()),
            )
            .map_err(crate::TantivyError::from)?;
            let doc: crate::TantivyDocument =
                crate::TantivyDocument::deserialize(deserializer)
                    .map_err(crate::TantivyError::from)?;
            let mut out: Vec<u8> = Vec::with_capacity(doc_bytes.len());
            let mut serializer = BinaryDocumentSerializer::new(&mut out, target_schema);
            serializer.serialize_doc(&doc)?;
            Ok(out)
        })
    }

    /// True if any block in this store has a non-identity remap trailer.
    /// Cheap pre-check for the merger's slow path: V1/V2 stores trivially
    /// return false, V3 stores scan block trailers (last 4 bytes per block).
    pub(crate) fn has_non_identity_remap(&self) -> bool {
        if self.doc_store_version < DocStoreVersion::V3 {
            return false;
        }
        for checkpoint in self.block_checkpoints() {
            let Ok(block_bytes) = self.get_compressed_block(&checkpoint) else {
                // I/O error here will surface again in the actual read; treat
                // as "might have remap" so the caller takes the safe slow path.
                return true;
            };
            let Ok((_, trailer_byte_len)) =
                super::block_trailer::read_block_trailer(block_bytes.as_ref())
            else {
                return true;
            };
            // 4 == empty trailer (just the u32 length itself).
            if trailer_byte_len > 4 {
                return true;
            }
        }
        false
    }

    /// Summarize total space usage of this store reader.
    pub fn space_usage(&self) -> StoreSpaceUsage {
        self.space_usage.clone()
    }
}

fn block_read_index(block: &[u8], doc_pos: u32) -> crate::Result<Range<usize>> {
    let doc_pos = doc_pos as usize;
    let size_of_u32 = std::mem::size_of::<u32>();

    let index_len_pos = block.len() - size_of_u32;
    let index_len = u32::deserialize(&mut &block[index_len_pos..])? as usize;

    if doc_pos > index_len {
        return Err(crate::TantivyError::InternalError(
            "Attempted to read doc from wrong block".to_owned(),
        ));
    }

    let index_start = block.len() - (index_len + 1) * size_of_u32;
    let index = &block[index_start..index_start + index_len * size_of_u32];

    let start_offset = u32::deserialize(&mut &index[doc_pos * size_of_u32..])? as usize;
    let end_offset = u32::deserialize(&mut &index[(doc_pos + 1) * size_of_u32..])
        .unwrap_or(index_start as u32) as usize;
    Ok(start_offset..end_offset)
}

#[cfg(feature = "quickwit")]
impl StoreReader {
    /// Advanced API.
    ///
    /// In most cases use [`get_async`](Self::get_async)
    ///
    /// Loads and decompresses a block asynchronously, stripping the V3
    /// per-block remap trailer (if any) before handing bytes to the
    /// decompressor.
    async fn read_block_async(
        &self,
        checkpoint: &Checkpoint,
        executor: &Executor,
    ) -> io::Result<Block> {
        let (block, _remap) = self.read_block_with_remap_async(checkpoint, executor).await?;
        Ok(block)
    }

    /// Async counterpart of [`read_block_with_remap`](Self::read_block_with_remap).
    /// Returns both the decompressed payload and any per-block field-id
    /// remap encoded in the V3 trailer; V1/V2 blocks always return an
    /// empty remap (identity).
    ///
    /// On a cache hit the FileSlice is not fetched at all — both payload
    /// and remap come from the LRU entry, matching the sync path.
    async fn read_block_with_remap_async(
        &self,
        checkpoint: &Checkpoint,
        executor: &Executor,
    ) -> io::Result<(Block, Arc<super::block_trailer::BlockFieldRemap>)> {
        let cache_key = checkpoint.byte_range.start;
        if let Some((cached_block, cached_remap)) = self.cache.get_from_cache(cache_key) {
            return Ok((cached_block, cached_remap));
        }

        let block_bytes = self
            .data
            .slice(checkpoint.byte_range.clone())
            .read_bytes_async()
            .await?;
        let (compressed_payload, remap) = self.split_block_trailer(block_bytes)?;

        let decompressor = self.decompressor;
        let maybe_decompressed_block = executor
            .spawn_blocking(move || decompressor.decompress(compressed_payload.as_ref()))
            .await
            .expect("decompression panicked");
        let decompressed_block = OwnedBytes::new(maybe_decompressed_block?);
        let remap_arc = Arc::new(remap);

        self.cache
            .put_into_cache(cache_key, decompressed_block.clone(), Arc::clone(&remap_arc));

        Ok((decompressed_block, remap_arc))
    }

    /// Reads raw bytes of a given document asynchronously.
    ///
    /// Note: for V3 blocks with a non-identity remap, these bytes encode the
    /// SOURCE schema's field ids and require trailer-aware decoding. Prefer
    /// [`get_async`](Self::get_async) for typed documents.
    pub async fn get_document_bytes_async(
        &self,
        doc_id: DocId,
        executor: &Executor,
    ) -> crate::Result<OwnedBytes> {
        let checkpoint = self.block_checkpoint(doc_id)?;
        let block = self.read_block_async(&checkpoint, executor).await?;
        Self::get_document_bytes_from_block(block, doc_id, &checkpoint)
    }

    /// Fetches a document asynchronously. Async version of [`get`](Self::get).
    pub async fn get_async<D: DocumentDeserialize>(
        &self,
        doc_id: DocId,
        executor: &Executor,
    ) -> crate::Result<D> {
        let checkpoint = self.block_checkpoint(doc_id)?;
        let (block, remap) = self.read_block_with_remap_async(&checkpoint, executor).await?;
        let mut doc_bytes = Self::get_document_bytes_from_block(block, doc_id, &checkpoint)?;
        let remap_ref = if remap.is_empty() { None } else { Some(remap.as_ref()) };
        let deserializer = BinaryDocumentDeserializer::from_reader_with_remap(
            &mut doc_bytes,
            self.doc_store_version,
            remap_ref,
        )
        .map_err(crate::TantivyError::from)?;
        D::deserialize(deserializer).map_err(crate::TantivyError::from)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::directory::RamDirectory;
    use crate::schema::{Field, TantivyDocument, Value};
    use crate::store::tests::write_lorem_ipsum_store;
    use crate::store::Compressor;
    use crate::Directory;

    const BLOCK_SIZE: usize = 16_384;

    fn get_text_field<'a>(doc: &'a TantivyDocument, field: &'a Field) -> Option<&'a str> {
        doc.get_first(*field).and_then(|f| f.as_value().as_str())
    }

    #[test]
    fn test_doc_store_version_ord() {
        assert!(DocStoreVersion::V1 < DocStoreVersion::V2);
    }

    #[test]
    fn test_store_lru_cache() -> crate::Result<()> {
        let directory = RamDirectory::create();
        let path = Path::new("store");
        let writer = directory.open_write(path)?;
        let schema = write_lorem_ipsum_store(writer, 500, Compressor::None, BLOCK_SIZE, true);
        let title = schema.get_field("title").unwrap();
        let store_file = directory.open_read(path)?;
        let store = StoreReader::open(store_file, DOCSTORE_CACHE_CAPACITY)?;

        assert_eq!(store.cache.len(), 0);
        assert_eq!(store.cache_stats().cache_hits, 0);
        assert_eq!(store.cache_stats().cache_misses, 0);

        let doc = store.get(0)?;
        assert_eq!(get_text_field(&doc, &title), Some("Doc 0"));

        assert_eq!(store.cache.len(), 1);
        assert_eq!(store.cache_stats().cache_hits, 0);
        assert_eq!(store.cache_stats().cache_misses, 1);

        assert_eq!(store.cache.peek_lru(), Some(0));

        let doc = store.get(499)?;
        assert_eq!(get_text_field(&doc, &title), Some("Doc 499"));

        assert_eq!(store.cache.len(), 2);
        assert_eq!(store.cache_stats().cache_hits, 0);
        assert_eq!(store.cache_stats().cache_misses, 2);

        assert_eq!(store.cache.peek_lru(), Some(0));

        let doc = store.get(0)?;
        assert_eq!(get_text_field(&doc, &title), Some("Doc 0"));

        assert_eq!(store.cache.len(), 2);
        assert_eq!(store.cache_stats().cache_hits, 1);
        assert_eq!(store.cache_stats().cache_misses, 2);

        // Each V3 block carries an extra 4-byte trailer (empty remap),
        // shifting every block's start offset by 4 bytes per preceding block.
        assert_eq!(store.cache.peek_lru(), Some(232262));

        Ok(())
    }
}
