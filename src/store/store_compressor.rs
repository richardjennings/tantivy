use std::io::Write;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::JoinHandle;
use std::{io, thread};

use common::{BinarySerializable, CountingWriter, TerminatingWrite};

use super::DOC_STORE_VERSION;
use crate::directory::WritePtr;
use crate::store::block_trailer::{write_block_trailer, BlockFieldRemap};
use crate::store::footer::DocStoreFooter;
use crate::store::index::{Checkpoint, SkipIndexBuilder};
use crate::store::reader::DocStoreVersion;
use crate::store::{Compressor, Decompressor, StoreReader};
use crate::DocId;

pub struct BlockCompressor(BlockCompressorVariants);

// The struct wrapping an enum is just here to keep the
// impls private.
enum BlockCompressorVariants {
    SameThread(BlockCompressorImpl),
    DedicatedThread(DedicatedThreadBlockCompressorImpl),
}

impl BlockCompressor {
    pub fn new(compressor: Compressor, wrt: WritePtr, dedicated_thread: bool) -> io::Result<Self> {
        let block_compressor_impl = BlockCompressorImpl::new(compressor, wrt);
        if dedicated_thread {
            let dedicated_thread_compressor =
                DedicatedThreadBlockCompressorImpl::new(block_compressor_impl)?;
            Ok(BlockCompressor(BlockCompressorVariants::DedicatedThread(
                dedicated_thread_compressor,
            )))
        } else {
            Ok(BlockCompressor(BlockCompressorVariants::SameThread(
                block_compressor_impl,
            )))
        }
    }

    pub fn compress_block_and_write(
        &mut self,
        bytes: &[u8],
        num_docs_in_block: u32,
    ) -> io::Result<()> {
        match &mut self.0 {
            BlockCompressorVariants::SameThread(block_compressor) => {
                block_compressor.compress_block_and_write(bytes, num_docs_in_block)?;
            }
            BlockCompressorVariants::DedicatedThread(different_thread_block_compressor) => {
                different_thread_block_compressor
                    .compress_block_and_write(bytes, num_docs_in_block)?;
            }
        }
        Ok(())
    }

    pub fn stack_reader(&mut self, store_reader: StoreReader) -> io::Result<()> {
        match &mut self.0 {
            BlockCompressorVariants::SameThread(block_compressor) => {
                block_compressor.stack(store_reader)?;
            }
            BlockCompressorVariants::DedicatedThread(different_thread_block_compressor) => {
                different_thread_block_compressor.stack_reader(store_reader)?;
            }
        }
        Ok(())
    }

    /// Stack a store reader with a per-block field-id remap applied. The
    /// compressed payload of every block is byte-copied through; only the
    /// V3 trailer is rewritten with the supplied remap. Used by the
    /// cross-schema re-segmenter to attach a translation header to source
    /// `.store` blocks without decompressing them.
    pub fn stack_reader_with_remap(
        &mut self,
        store_reader: StoreReader,
        remap: BlockFieldRemap,
    ) -> io::Result<()> {
        match &mut self.0 {
            BlockCompressorVariants::SameThread(block_compressor) => {
                block_compressor.stack_with_remap(store_reader, remap)?;
            }
            BlockCompressorVariants::DedicatedThread(different_thread_block_compressor) => {
                different_thread_block_compressor
                    .stack_reader_with_remap(store_reader, remap)?;
            }
        }
        Ok(())
    }

    pub fn close(self) -> io::Result<()> {
        let imp = self.0;
        match imp {
            BlockCompressorVariants::SameThread(block_compressor) => block_compressor.close(),
            BlockCompressorVariants::DedicatedThread(different_thread_block_compressor) => {
                different_thread_block_compressor.close()
            }
        }
    }
}

struct BlockCompressorImpl {
    compressor: Compressor,
    first_doc_in_block: DocId,
    offset_index_writer: SkipIndexBuilder,
    intermediary_buffer: Vec<u8>,
    writer: CountingWriter<WritePtr>,
}

impl BlockCompressorImpl {
    fn new(compressor: Compressor, writer: WritePtr) -> Self {
        Self {
            compressor,
            first_doc_in_block: 0,
            offset_index_writer: SkipIndexBuilder::new(),
            intermediary_buffer: Vec::new(),
            writer: CountingWriter::wrap(writer),
        }
    }

    fn compress_block_and_write(&mut self, data: &[u8], num_docs_in_block: u32) -> io::Result<()> {
        assert!(num_docs_in_block > 0);
        self.intermediary_buffer.clear();
        self.compressor
            .compress_into(data, &mut self.intermediary_buffer)?;

        let start_offset = self.writer.written_bytes() as usize;
        self.writer.write_all(&self.intermediary_buffer)?;
        // V3 blocks carry a per-block remap trailer right after the compressed
        // payload. For freshly-written blocks the remap is empty (identity);
        // the trailer occupies exactly 4 bytes (`u32 trailer_byte_len = 4`).
        // Older V2 blocks (e.g. read by an upgraded reader) have no trailer
        // — version dispatch handles that on the read side.
        if DOC_STORE_VERSION >= DocStoreVersion::V3 {
            write_block_trailer(&mut self.writer, &BlockFieldRemap::default())?;
        }
        let end_offset = self.writer.written_bytes() as usize;

        self.register_checkpoint(Checkpoint {
            doc_range: self.first_doc_in_block..self.first_doc_in_block + num_docs_in_block,
            byte_range: start_offset..end_offset,
        });
        Ok(())
    }

    fn register_checkpoint(&mut self, checkpoint: Checkpoint) {
        self.offset_index_writer.insert(checkpoint.clone());
        self.first_doc_in_block = checkpoint.doc_range.end;
    }

    /// Stacks a store reader on top of the documents written so far.
    /// This method is an optimization compared to iterating over the documents
    /// in the store and adding them one by one, as the store's data will
    /// not be decompressed and then recompressed.
    fn stack(&mut self, store_reader: StoreReader) -> io::Result<()> {
        let doc_shift = self.first_doc_in_block;
        let start_shift = self.writer.written_bytes() as usize;

        // just bulk write all of the block of the given reader.
        self.writer
            .write_all(store_reader.block_data()?.as_slice())?;

        // concatenate the index of the `store_reader`, after translating
        // its start doc id and its start file offset.
        for mut checkpoint in store_reader.block_checkpoints() {
            checkpoint.doc_range.start += doc_shift;
            checkpoint.doc_range.end += doc_shift;
            checkpoint.byte_range.start += start_shift;
            checkpoint.byte_range.end += start_shift;
            self.register_checkpoint(checkpoint);
        }
        Ok(())
    }

    /// Stack with per-block field-id remap. Walks each source block,
    /// strips its trailer (if any), byte-copies the compressed payload to
    /// the target, then writes a fresh trailer carrying the composed
    /// remap. No decompression, no recompression.
    fn stack_with_remap(
        &mut self,
        store_reader: StoreReader,
        user_remap: BlockFieldRemap,
    ) -> io::Result<()> {
        let doc_shift = self.first_doc_in_block;
        let source_block_data = store_reader.block_data()?;
        let source_version = store_reader.doc_store_version();
        let source_checkpoints: Vec<Checkpoint> = store_reader.block_checkpoints().collect();

        for checkpoint in source_checkpoints {
            let block_bytes = source_block_data.slice(checkpoint.byte_range.clone());

            // Strip the source's V3 trailer (if any) to recover the compressed
            // payload. For V2 sources there's no trailer; the whole block is
            // the payload and the source remap is empty/identity.
            let (compressed_payload, source_remap) =
                if source_version >= DocStoreVersion::V3 {
                    let (remap, trailer_len) =
                        crate::store::read_block_trailer(block_bytes.as_ref())?;
                    let payload_end = block_bytes.len() - trailer_len;
                    (block_bytes.slice(0..payload_end), remap)
                } else {
                    (block_bytes, BlockFieldRemap::default())
                };

            // Compose the source's remap with the user's translation:
            //   composed[encoded] = user_remap[ source_remap[encoded] ]
            //
            // For freshly-written V3 sources the source_remap is empty and
            // composition is just `composed = user_remap` over the source
            // schema's field ids. We replicate the user_remap entries
            // directly here — any encoded id not in the user_remap will
            // fall through to identity at read time, which matches
            // "field not in field_map → keep its id" semantics.
            let mut composed_pairs: Vec<(u32, u32)> = Vec::new();
            if source_remap.is_empty() {
                for (encoded, target) in user_remap.pairs() {
                    composed_pairs.push((encoded, target));
                }
            } else {
                for (encoded, source_id) in source_remap.pairs() {
                    let target = user_remap.lookup(source_id);
                    composed_pairs.push((encoded, target));
                }
                // Also include user_remap entries that don't appear in
                // source_remap — they may still be encoded as identity in
                // the payload.
                for (encoded, target) in user_remap.pairs() {
                    if source_remap.lookup(encoded) == encoded {
                        composed_pairs.push((encoded, target));
                    }
                }
            }
            let composed = BlockFieldRemap::from_pairs(composed_pairs);

            // Write the (untouched) compressed payload, then the composed
            // V3 trailer. Record the new checkpoint at the resulting byte
            // range and doc range.
            let start_offset = self.writer.written_bytes() as usize;
            self.writer.write_all(compressed_payload.as_slice())?;
            crate::store::write_block_trailer(&mut self.writer, &composed)?;
            let end_offset = self.writer.written_bytes() as usize;

            self.register_checkpoint(Checkpoint {
                doc_range: (checkpoint.doc_range.start + doc_shift)
                    ..(checkpoint.doc_range.end + doc_shift),
                byte_range: start_offset..end_offset,
            });
        }
        Ok(())
    }

    fn close(mut self) -> io::Result<()> {
        let header_offset: u64 = self.writer.written_bytes();
        let docstore_footer = DocStoreFooter::new(
            header_offset,
            Decompressor::from(self.compressor),
            DOC_STORE_VERSION,
        );
        self.offset_index_writer.serialize_into(&mut self.writer)?;
        docstore_footer.serialize(&mut self.writer)?;
        self.writer.terminate()
    }
}

// ---------------------------------
enum BlockCompressorMessage {
    CompressBlockAndWrite {
        block_data: Vec<u8>,
        num_docs_in_block: u32,
    },
    Stack(StoreReader),
    StackWithRemap(StoreReader, BlockFieldRemap),
}

struct DedicatedThreadBlockCompressorImpl {
    join_handle: Option<JoinHandle<io::Result<()>>>,
    tx: SyncSender<BlockCompressorMessage>,
}

impl DedicatedThreadBlockCompressorImpl {
    fn new(mut block_compressor: BlockCompressorImpl) -> io::Result<Self> {
        let (tx, rx): (
            SyncSender<BlockCompressorMessage>,
            Receiver<BlockCompressorMessage>,
        ) = sync_channel(3);
        let join_handle = thread::Builder::new()
            .name("docstore-compressor-thread".to_string())
            .spawn(move || {
                while let Ok(packet) = rx.recv() {
                    match packet {
                        BlockCompressorMessage::CompressBlockAndWrite {
                            block_data,
                            num_docs_in_block,
                        } => {
                            block_compressor
                                .compress_block_and_write(&block_data[..], num_docs_in_block)?;
                        }
                        BlockCompressorMessage::Stack(store_reader) => {
                            block_compressor.stack(store_reader)?;
                        }
                        BlockCompressorMessage::StackWithRemap(store_reader, remap) => {
                            block_compressor.stack_with_remap(store_reader, remap)?;
                        }
                    }
                }
                block_compressor.close()?;
                Ok(())
            })?;
        Ok(DedicatedThreadBlockCompressorImpl {
            join_handle: Some(join_handle),
            tx,
        })
    }

    fn compress_block_and_write(&mut self, bytes: &[u8], num_docs_in_block: u32) -> io::Result<()> {
        self.send(BlockCompressorMessage::CompressBlockAndWrite {
            block_data: bytes.to_vec(),
            num_docs_in_block,
        })
    }

    fn stack_reader(&mut self, store_reader: StoreReader) -> io::Result<()> {
        self.send(BlockCompressorMessage::Stack(store_reader))
    }

    fn stack_reader_with_remap(
        &mut self,
        store_reader: StoreReader,
        remap: BlockFieldRemap,
    ) -> io::Result<()> {
        self.send(BlockCompressorMessage::StackWithRemap(store_reader, remap))
    }

    fn send(&mut self, msg: BlockCompressorMessage) -> io::Result<()> {
        if self.tx.send(msg).is_err() {
            harvest_thread_result(self.join_handle.take())?;
            return Err(io::Error::other("Unidentified error."));
        }
        Ok(())
    }

    fn close(self) -> io::Result<()> {
        drop(self.tx);
        harvest_thread_result(self.join_handle)
    }
}

/// Wait for the thread result to terminate and returns its result.
///
/// If the thread panicked, or if the result has already been harvested,
/// returns an explicit error.
fn harvest_thread_result(join_handle_opt: Option<JoinHandle<io::Result<()>>>) -> io::Result<()> {
    let join_handle = join_handle_opt.ok_or_else(|| io::Error::other("Thread already joined."))?;
    join_handle
        .join()
        .map_err(|_err| io::Error::other("Compressing thread panicked."))?
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::path::Path;

    use crate::directory::RamDirectory;
    use crate::store::store_compressor::BlockCompressor;
    use crate::store::Compressor;
    use crate::Directory;

    fn populate_block_compressor(mut block_compressor: BlockCompressor) -> io::Result<()> {
        block_compressor.compress_block_and_write(b"hello", 1)?;
        block_compressor.compress_block_and_write(b"happy", 1)?;
        block_compressor.close()?;
        Ok(())
    }

    #[test]
    fn test_block_store_compressor_impls_yield_the_same_result() {
        let ram_directory = RamDirectory::default();
        let path1 = Path::new("path1");
        let path2 = Path::new("path2");
        let wrt1 = ram_directory.open_write(path1).unwrap();
        let wrt2 = ram_directory.open_write(path2).unwrap();
        let block_compressor1 = BlockCompressor::new(Compressor::None, wrt1, true).unwrap();
        let block_compressor2 = BlockCompressor::new(Compressor::None, wrt2, false).unwrap();
        populate_block_compressor(block_compressor1).unwrap();
        populate_block_compressor(block_compressor2).unwrap();
        let data1 = ram_directory.open_read(path1).unwrap();
        let data2 = ram_directory.open_read(path2).unwrap();
        assert_eq!(data1.read_bytes().unwrap(), data2.read_bytes().unwrap());
    }
}
