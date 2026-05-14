use std::collections::HashSet;
use std::fmt;
#[cfg(feature = "mmap")]
use std::path::Path;
use std::path::PathBuf;
use std::thread::available_parallelism;

use super::segment::Segment;
use super::segment_reader::merge_field_meta_data;
use super::{FieldMetadata, IndexSettings};
use crate::core::{Executor, META_FILEPATH};
use crate::directory::error::OpenReadError;
#[cfg(feature = "mmap")]
use crate::directory::MmapDirectory;
use crate::directory::{Directory, ManagedDirectory, RamDirectory, INDEX_WRITER_LOCK};
use crate::error::{DataCorruption, TantivyError};
use crate::index::{IndexMeta, SegmentId, SegmentMeta, SegmentMetaInventory};
use crate::indexer::index_writer::{
    IndexWriterOptions, MAX_NUM_THREAD, MEMORY_BUDGET_NUM_BYTES_MIN,
};
use crate::indexer::segment_updater::save_metas;
use crate::indexer::{IndexWriter, SingleSegmentIndexWriter};
use crate::reader::{IndexReader, IndexReaderBuilder};
use crate::schema::document::Document;
use crate::schema::{Field, FieldType, Schema};
use crate::tokenizer::{TextAnalyzer, TokenizerManager};
use crate::SegmentReader;

fn load_metas(
    directory: &dyn Directory,
    inventory: &SegmentMetaInventory,
) -> crate::Result<IndexMeta> {
    let meta_data = directory.atomic_read(&META_FILEPATH)?;
    let meta_string = String::from_utf8(meta_data).map_err(|_utf8_err| {
        error!("Meta data is not valid utf8.");
        DataCorruption::new(
            META_FILEPATH.to_path_buf(),
            "Meta file does not contain valid utf8 file.".to_string(),
        )
    })?;
    IndexMeta::deserialize(&meta_string, inventory)
        .map_err(|e| {
            DataCorruption::new(
                META_FILEPATH.to_path_buf(),
                format!("Meta file cannot be deserialized. {e:?}. Content: {meta_string:?}"),
            )
        })
        .map_err(From::from)
}

/// Save the index meta file.
/// This operation is atomic :
/// Either
/// - it fails, in which case an error is returned, and the `meta.json` remains untouched,
/// - it succeeds, and `meta.json` is written and flushed.
///
/// This method is not part of tantivy's public API
fn save_new_metas(
    schema: Schema,
    index_settings: IndexSettings,
    directory: &dyn Directory,
) -> crate::Result<()> {
    save_metas(
        &IndexMeta {
            index_settings,
            segments: Vec::new(),
            schema,
            opstamp: 0u64,
            payload: None,
        },
        directory,
    )?;
    directory.sync_directory()?;
    Ok(())
}

/// IndexBuilder can be used to create an index.
///
/// Use in conjunction with [`SchemaBuilder`][crate::schema::SchemaBuilder].
/// Global index settings can be configured with [`IndexSettings`].
///
/// # Examples
///
/// ```
/// use tantivy::schema::*;
/// use tantivy::{Index, IndexSettings};
///
/// let mut schema_builder = Schema::builder();
/// let id_field = schema_builder.add_text_field("id", STRING);
/// let title_field = schema_builder.add_text_field("title", TEXT);
/// let body_field = schema_builder.add_text_field("body", TEXT);
/// let number_field = schema_builder.add_u64_field(
///     "number",
///     NumericOptions::default().set_fast(),
/// );
///
/// let schema = schema_builder.build();
/// let settings = IndexSettings{
///     docstore_blocksize: 100_000,
///     ..Default::default()
/// };
/// let index = Index::builder().schema(schema).settings(settings).create_in_ram();
/// ```
pub struct IndexBuilder {
    schema: Option<Schema>,
    index_settings: IndexSettings,
    tokenizer_manager: TokenizerManager,
    fast_field_tokenizer_manager: TokenizerManager,
}
impl Default for IndexBuilder {
    fn default() -> Self {
        IndexBuilder::new()
    }
}
impl IndexBuilder {
    /// Creates a new `IndexBuilder`
    pub fn new() -> Self {
        Self {
            schema: None,
            index_settings: IndexSettings::default(),
            tokenizer_manager: TokenizerManager::default(),
            fast_field_tokenizer_manager: TokenizerManager::default(),
        }
    }

    /// Set the settings
    #[must_use]
    pub fn settings(mut self, settings: IndexSettings) -> Self {
        self.index_settings = settings;
        self
    }

    /// Set the schema
    #[must_use]
    pub fn schema(mut self, schema: Schema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Set the tokenizers.
    pub fn tokenizers(mut self, tokenizers: TokenizerManager) -> Self {
        self.tokenizer_manager = tokenizers;
        self
    }

    /// Set the fast field tokenizers.
    pub fn fast_field_tokenizers(mut self, tokenizers: TokenizerManager) -> Self {
        self.fast_field_tokenizer_manager = tokenizers;
        self
    }

    /// Creates a new index using the [`RamDirectory`].
    ///
    /// The index will be allocated in anonymous memory.
    /// This is useful for indexing small set of documents
    /// for instances like unit test or temporary in memory index.
    pub fn create_in_ram(self) -> Result<Index, TantivyError> {
        let ram_directory = RamDirectory::create();
        self.create(ram_directory)
    }

    /// Creates a new index in a given filepath.
    /// The index will use the [`MmapDirectory`].
    ///
    /// If a previous index was in this directory, it returns an
    /// [`TantivyError::IndexAlreadyExists`] error.
    #[cfg(feature = "mmap")]
    pub fn create_in_dir<P: AsRef<Path>>(self, directory_path: P) -> crate::Result<Index> {
        let mmap_directory: Box<dyn Directory> = Box::new(MmapDirectory::open(directory_path)?);
        if Index::exists(&*mmap_directory)? {
            return Err(TantivyError::IndexAlreadyExists);
        }
        self.create(mmap_directory)
    }

    /// Dragons ahead!!!
    ///
    /// The point of this API is to let users create a simple index with a single segment
    /// and without starting any thread.
    ///
    /// Do not use this method if you are not sure what you are doing.
    ///
    /// It expects an originally empty directory, and will not run any GC operation.
    #[doc(hidden)]
    pub fn single_segment_index_writer<D: Document>(
        self,
        dir: impl Into<Box<dyn Directory>>,
        mem_budget: usize,
    ) -> crate::Result<SingleSegmentIndexWriter<D>> {
        let index = self.create(dir)?;
        let index_simple_writer = SingleSegmentIndexWriter::new(index, mem_budget)?;
        Ok(index_simple_writer)
    }

    /// Creates a new index in a temp directory.
    ///
    /// The index will use the [`MmapDirectory`] in a newly created directory.
    /// The temp directory will be destroyed automatically when the [`Index`] object
    /// is destroyed.
    ///
    /// The temp directory is only used for testing the [`MmapDirectory`].
    /// For other unit tests, prefer the [`RamDirectory`], see:
    /// [`IndexBuilder::create_in_ram()`].
    #[cfg(feature = "mmap")]
    pub fn create_from_tempdir(self) -> crate::Result<Index> {
        let mmap_directory: Box<dyn Directory> = Box::new(MmapDirectory::create_from_tempdir()?);
        self.create(mmap_directory)
    }

    fn get_expect_schema(&self) -> crate::Result<Schema> {
        self.schema
            .as_ref()
            .cloned()
            .ok_or(TantivyError::IndexBuilderMissingArgument("schema"))
    }

    /// Opens or creates a new index in the provided directory
    pub fn open_or_create<T: Into<Box<dyn Directory>>>(self, dir: T) -> crate::Result<Index> {
        let dir: Box<dyn Directory> = dir.into();
        if !Index::exists(&*dir)? {
            return self.create(dir);
        }
        let mut index = Index::open(dir)?;
        index.set_tokenizers(self.tokenizer_manager.clone());
        if index.schema() == self.get_expect_schema()? {
            Ok(index)
        } else {
            Err(TantivyError::SchemaError(
                "An index exists but the schema does not match.".to_string(),
            ))
        }
    }

    fn validate(&self) -> crate::Result<()> {
        if let Some(_schema) = self.schema.as_ref() {
            Ok(())
        } else {
            Err(TantivyError::InvalidArgument(
                "no schema passed".to_string(),
            ))
        }
    }

    /// Creates a new index given an implementation of the trait `Directory`.
    ///
    /// If a directory previously existed, it will be erased.
    fn create<T: Into<Box<dyn Directory>>>(self, dir: T) -> crate::Result<Index> {
        self.validate()?;
        let dir = dir.into();
        let directory = ManagedDirectory::wrap(dir)?;
        save_new_metas(
            self.get_expect_schema()?,
            self.index_settings.clone(),
            &directory,
        )?;
        let mut metas = IndexMeta::with_schema(self.get_expect_schema()?);
        metas.index_settings = self.index_settings;
        let mut index = Index::open_from_metas(directory, &metas, SegmentMetaInventory::default());
        index.set_tokenizers(self.tokenizer_manager);
        index.set_fast_field_tokenizers(self.fast_field_tokenizer_manager);
        Ok(index)
    }
}

/// Search Index
#[derive(Clone)]
pub struct Index {
    directory: ManagedDirectory,
    schema: Schema,
    settings: IndexSettings,
    executor: Executor,
    tokenizers: TokenizerManager,
    fast_field_tokenizers: TokenizerManager,
    inventory: SegmentMetaInventory,
}

impl Index {
    /// Creates a new builder.
    pub fn builder() -> IndexBuilder {
        IndexBuilder::new()
    }
    /// Examines the directory to see if it contains an index.
    ///
    /// Effectively, it only checks for the presence of the `meta.json` file.
    pub fn exists(dir: &dyn Directory) -> Result<bool, OpenReadError> {
        dir.exists(&META_FILEPATH)
    }

    /// Accessor to the search executor.
    ///
    /// This pool is used by default when calling `searcher.search(...)`
    /// to perform search on the individual segments.
    ///
    /// By default the executor is single thread, and simply runs in the calling thread.
    pub fn search_executor(&self) -> &Executor {
        &self.executor
    }

    /// Replace the default single thread search executor pool
    /// by a thread pool with a given number of threads.
    pub fn set_multithread_executor(&mut self, num_threads: usize) -> crate::Result<()> {
        self.executor = Executor::multi_thread(num_threads, "tantivy-search-")?;
        Ok(())
    }

    /// Custom thread pool by a outer thread pool.
    pub fn set_executor(&mut self, executor: Executor) {
        self.executor = executor;
    }

    /// Replace the default single thread search executor pool
    /// by a thread pool with as many threads as there are CPUs on the system.
    pub fn set_default_multithread_executor(&mut self) -> crate::Result<()> {
        let default_num_threads = available_parallelism()?.get();
        self.set_multithread_executor(default_num_threads)
    }

    /// Creates a new index using the [`RamDirectory`].
    ///
    /// The index will be allocated in anonymous memory.
    /// This is useful for indexing small set of documents
    /// for instances like unit test or temporary in memory index.
    pub fn create_in_ram(schema: Schema) -> Index {
        IndexBuilder::new().schema(schema).create_in_ram().unwrap()
    }

    /// Creates a new index in a given filepath.
    /// The index will use the [`MmapDirectory`].
    ///
    /// If a previous index was in this directory, then it returns
    /// a [`TantivyError::IndexAlreadyExists`] error.
    #[cfg(feature = "mmap")]
    pub fn create_in_dir<P: AsRef<Path>>(
        directory_path: P,
        schema: Schema,
    ) -> crate::Result<Index> {
        IndexBuilder::new()
            .schema(schema)
            .create_in_dir(directory_path)
    }

    /// Opens or creates a new index in the provided directory
    pub fn open_or_create<T: Into<Box<dyn Directory>>>(
        dir: T,
        schema: Schema,
    ) -> crate::Result<Index> {
        let dir = dir.into();
        IndexBuilder::new().schema(schema).open_or_create(dir)
    }

    /// Creates a new index in a temp directory.
    ///
    /// The index will use the [`MmapDirectory`] in a newly created directory.
    /// The temp directory will be destroyed automatically when the [`Index`] object
    /// is destroyed.
    ///
    /// The temp directory is only used for testing the [`MmapDirectory`].
    /// For other unit tests, prefer the [`RamDirectory`],
    /// see: [`IndexBuilder::create_in_ram()`].
    #[cfg(feature = "mmap")]
    pub fn create_from_tempdir(schema: Schema) -> crate::Result<Index> {
        IndexBuilder::new().schema(schema).create_from_tempdir()
    }

    /// Creates a new index given an implementation of the trait `Directory`.
    ///
    /// If a directory previously existed, it will be erased.
    pub fn create<T: Into<Box<dyn Directory>>>(
        dir: T,
        schema: Schema,
        settings: IndexSettings,
    ) -> crate::Result<Index> {
        let dir: Box<dyn Directory> = dir.into();
        let mut builder = IndexBuilder::new().schema(schema);
        builder = builder.settings(settings);
        builder.create(dir)
    }

    /// Creates a new index given a directory and an [`IndexMeta`].
    fn open_from_metas(
        directory: ManagedDirectory,
        metas: &IndexMeta,
        inventory: SegmentMetaInventory,
    ) -> Index {
        let schema = metas.schema.clone();
        Index {
            settings: metas.index_settings.clone(),
            directory,
            schema,
            tokenizers: TokenizerManager::default(),
            fast_field_tokenizers: TokenizerManager::default(),
            executor: Executor::single_thread(),
            inventory,
        }
    }

    /// Setter for the tokenizer manager.
    pub fn set_tokenizers(&mut self, tokenizers: TokenizerManager) {
        self.tokenizers = tokenizers;
    }

    /// Accessor for the tokenizer manager.
    pub fn tokenizers(&self) -> &TokenizerManager {
        &self.tokenizers
    }

    /// Setter for the fast field tokenizer manager.
    pub fn set_fast_field_tokenizers(&mut self, tokenizers: TokenizerManager) {
        self.fast_field_tokenizers = tokenizers;
    }

    /// Accessor for the fast field tokenizer manager.
    pub fn fast_field_tokenizer(&self) -> &TokenizerManager {
        &self.fast_field_tokenizers
    }

    /// Get the tokenizer associated with a specific field.
    pub fn tokenizer_for_field(&self, field: Field) -> crate::Result<TextAnalyzer> {
        let field_entry = self.schema.get_field_entry(field);
        let field_type = field_entry.field_type();
        let tokenizer_manager: &TokenizerManager = self.tokenizers();
        let indexing_options_opt = match field_type {
            FieldType::JsonObject(options) => options.get_text_indexing_options(),
            FieldType::Str(options) => options.get_indexing_options(),
            _ => {
                return Err(TantivyError::SchemaError(format!(
                    "{:?} is not a text field.",
                    field_entry.name()
                )))
            }
        };
        let indexing_options = indexing_options_opt.ok_or_else(|| {
            TantivyError::InvalidArgument(format!(
                "No indexing options set for field {field_entry:?}"
            ))
        })?;

        tokenizer_manager
            .get(indexing_options.tokenizer())
            .ok_or_else(|| {
                TantivyError::InvalidArgument(format!(
                    "No Tokenizer found for field {field_entry:?}"
                ))
            })
    }

    /// Create a default [`IndexReader`] for the given index.
    ///
    /// See [`Index.reader_builder()`].
    pub fn reader(&self) -> crate::Result<IndexReader> {
        self.reader_builder().try_into()
    }

    /// Create a [`IndexReader`] for the given index.
    ///
    /// Most project should create at most one reader for a given index.
    /// This method is typically called only once per `Index` instance.
    pub fn reader_builder(&self) -> IndexReaderBuilder {
        IndexReaderBuilder::new(self.clone())
    }

    /// Opens a new directory from an index path.
    #[cfg(feature = "mmap")]
    pub fn open_in_dir<P: AsRef<Path>>(directory_path: P) -> crate::Result<Index> {
        let mmap_directory = MmapDirectory::open(directory_path)?;
        Index::open(mmap_directory)
    }

    /// Returns the list of the segment metas tracked by the index.
    ///
    /// Such segments can of course be part of the index,
    /// but also they could be segments being currently built or in the middle of a merge
    /// operation.
    pub(crate) fn list_all_segment_metas(&self) -> Vec<SegmentMeta> {
        self.inventory.all()
    }

    /// Returns the list of fields that have been indexed in the Index.
    /// The field list includes the field defined in the schema as well as the fields
    /// that have been indexed as a part of a JSON field.
    /// The returned field name is the full field name, including the name of the JSON field.
    ///
    /// The returned field names can be used in queries.
    ///
    /// Notice: If your data contains JSON fields this is **very expensive**, as it requires
    /// browsing through the inverted index term dictionary and the columnar field dictionary.
    ///
    /// Disclaimer: Some fields may not be listed here. For instance, if the schema contains a json
    /// field that is not indexed nor a fast field but is stored, it is possible for the field
    /// to not be listed.
    pub fn fields_metadata(&self) -> crate::Result<Vec<FieldMetadata>> {
        let segments = self.searchable_segments()?;
        let fields_metadata: Vec<Vec<FieldMetadata>> = segments
            .into_iter()
            .map(|segment| SegmentReader::open(&segment)?.fields_metadata())
            .collect::<Result<_, _>>()?;
        Ok(merge_field_meta_data(fields_metadata))
    }

    /// Creates a new segment_meta (Advanced user only).
    ///
    /// As long as the `SegmentMeta` lives, the files associated with the
    /// `SegmentMeta` are guaranteed to not be garbage collected, regardless of
    /// whether the segment is recorded as part of the index or not.
    pub fn new_segment_meta(&self, segment_id: SegmentId, max_doc: u32) -> SegmentMeta {
        self.inventory.new_segment_meta(segment_id, max_doc)
    }

    /// Open the index using the provided directory
    pub fn open<T: Into<Box<dyn Directory>>>(directory: T) -> crate::Result<Index> {
        let directory = directory.into();
        let directory = ManagedDirectory::wrap(directory)?;
        let inventory = SegmentMetaInventory::default();
        let metas = load_metas(&directory, &inventory)?;
        let index = Index::open_from_metas(directory, &metas, inventory);
        Ok(index)
    }

    /// Reads the index meta file from the directory.
    pub fn load_metas(&self) -> crate::Result<IndexMeta> {
        load_metas(self.directory(), &self.inventory)
    }

    /// Open a new index writer with the given options. Attempts to acquire a lockfile.
    ///
    /// The lockfile should be deleted on drop, but it is possible
    /// that due to a panic or other error, a stale lockfile will be
    /// left in the index directory. If you are sure that no other
    /// `IndexWriter` on the system is accessing the index directory,
    /// it is safe to manually delete the lockfile.
    ///
    /// - `options` defines the writer configuration which includes things like buffer sizes,
    ///   indexer threads, etc...
    ///
    /// # Errors
    /// If the lockfile already exists, returns `TantivyError::LockFailure`.
    /// If the memory arena per thread is too small or too big, returns
    /// `TantivyError::InvalidArgument`
    pub fn writer_with_options<D: Document>(
        &self,
        options: IndexWriterOptions,
    ) -> crate::Result<IndexWriter<D>> {
        let directory_lock = self
            .directory
            .acquire_lock(&INDEX_WRITER_LOCK)
            .map_err(|err| {
                TantivyError::LockFailure(
                    err,
                    Some(
                        "Failed to acquire index lock. If you are using a regular directory, this \
                         means there is already an `IndexWriter` working on this `Directory`, in \
                         this process or in a different process."
                            .to_string(),
                    ),
                )
            })?;

        IndexWriter::new(self, options, directory_lock)
    }

    /// Open a new index writer. Attempts to acquire a lockfile.
    ///
    /// The lockfile should be deleted on drop, but it is possible
    /// that due to a panic or other error, a stale lockfile will be
    /// left in the index directory. If you are sure that no other
    /// `IndexWriter` on the system is accessing the index directory,
    /// it is safe to manually delete the lockfile.
    ///
    /// - `num_threads` defines the number of indexing workers that should work at the same time.
    ///
    /// - `overall_memory_budget_in_bytes` sets the amount of memory allocated for all indexing
    ///   thread.
    ///
    /// Each thread will receive a budget of `overall_memory_budget_in_bytes / num_threads`.
    ///
    /// # Errors
    /// If the lockfile already exists, returns `Error::DirectoryLockBusy` or an `Error::IoError`.
    /// If the memory arena per thread is too small or too big, returns
    /// `TantivyError::InvalidArgument`
    pub fn writer_with_num_threads<D: Document>(
        &self,
        num_threads: usize,
        overall_memory_budget_in_bytes: usize,
    ) -> crate::Result<IndexWriter<D>> {
        let memory_arena_in_bytes_per_thread = overall_memory_budget_in_bytes / num_threads;
        let options = IndexWriterOptions::builder()
            .num_worker_threads(num_threads)
            .memory_budget_per_thread(memory_arena_in_bytes_per_thread)
            .build();
        self.writer_with_options(options)
    }

    /// Helper to create an index writer for tests.
    ///
    /// That index writer only simply has a single thread and a memory budget of 15 MB.
    /// Using a single thread gives us a deterministic allocation of DocId.
    #[cfg(test)]
    pub fn writer_for_tests<D: Document>(&self) -> crate::Result<IndexWriter<D>> {
        self.writer_with_num_threads(1, MEMORY_BUDGET_NUM_BYTES_MIN)
    }

    /// Creates a multithreaded writer
    ///
    /// Tantivy will automatically define the number of threads to use, but
    /// no more than 8 threads.
    /// `overall_memory_arena_in_bytes` is the total target memory usage that will be split
    /// between a given number of threads.
    ///
    /// # Errors
    /// If the lockfile already exists, returns `Error::FileAlreadyExists`.
    /// If the memory arena per thread is too small or too big, returns
    /// `TantivyError::InvalidArgument`
    pub fn writer<D: Document>(
        &self,
        memory_budget_in_bytes: usize,
    ) -> crate::Result<IndexWriter<D>> {
        let mut num_threads = std::cmp::min(available_parallelism()?.get(), MAX_NUM_THREAD);
        let memory_budget_num_bytes_per_thread = memory_budget_in_bytes / num_threads;
        if memory_budget_num_bytes_per_thread < MEMORY_BUDGET_NUM_BYTES_MIN {
            num_threads = (memory_budget_in_bytes / MEMORY_BUDGET_NUM_BYTES_MIN).max(1);
        }
        self.writer_with_num_threads(num_threads, memory_budget_in_bytes)
    }

    /// Accessor to the index settings
    pub fn settings(&self) -> &IndexSettings {
        &self.settings
    }

    /// Accessor to the index settings
    pub fn settings_mut(&mut self) -> &mut IndexSettings {
        &mut self.settings
    }

    /// Accessor to the index schema
    ///
    /// The schema is actually cloned.
    pub fn schema(&self) -> Schema {
        self.schema.clone()
    }

    /// Extend the index's schema with additional fields appended at the end.
    ///
    /// `new_schema` MUST be a strict prefix extension of the current schema:
    /// every field that was in the current schema must appear in
    /// `new_schema` at the same position, with the same name, type, and
    /// options. New fields may be appended after the existing ones (they
    /// get new Field IDs assigned by `SchemaBuilder` in the usual way).
    ///
    /// This is the on-disk side of additive schema evolution. Existing
    /// segments stay valid: their stored docs encode the old Field IDs,
    /// which remain correctly positioned under the extended schema. The
    /// segments are simply *sparse* on the new fields — exactly the same
    /// semantics tantivy uses for any doc that doesn't set a value.
    ///
    /// The new schema is persisted to `meta.json` atomically. Any
    /// `IndexWriter` / `IndexReader` opened before the extension still
    /// holds a clone of the previous schema and must be re-opened to see
    /// the new fields.
    ///
    /// Returns an error and leaves the on-disk state untouched if
    /// `new_schema` is not a valid prefix extension.
    /// Internal: overwrite this `Index` clone's schema without touching
    /// disk. Used by `IndexWriter::extend_schema` to propagate a freshly
    /// validated schema to the `SegmentUpdater`'s own `Index` clone so its
    /// subsequent `new_segment` calls (in the merge thread) see the new
    /// fields. `extend_schema` itself does the validation + meta.json
    /// rewrite; this is just the in-memory swap.
    pub(crate) fn set_schema_in_memory(&mut self, new_schema: Schema) {
        self.schema = new_schema;
    }

    pub fn extend_schema(&mut self, new_schema: Schema) -> crate::Result<()> {
        // Acquire the same exclusive lock that `IndexWriter` creation takes,
        // so a concurrent commit or end-merge in another process can't
        // interleave with the meta.json read-modify-write below and clobber
        // either change. Held for the duration of the load_metas + save_metas
        // pair and dropped at function exit.
        let _writer_lock = self
            .directory
            .acquire_lock(&INDEX_WRITER_LOCK)
            .map_err(|err| {
                TantivyError::LockFailure(
                    err,
                    Some(
                        "extend_schema: could not acquire INDEX_WRITER_LOCK \
                         — another IndexWriter is open on this directory."
                            .to_string(),
                    ),
                )
            })?;
        self.extend_schema_no_lock(new_schema)
    }

    /// Internal counterpart to [`Index::extend_schema`] that skips the
    /// directory lock acquisition. Used by [`crate::IndexWriter::extend_schema`]
    /// because the writer already holds INDEX_WRITER_LOCK for its lifetime —
    /// reacquiring would deadlock or fail with "lock already held".
    pub(crate) fn extend_schema_no_lock(&mut self, new_schema: Schema) -> crate::Result<()> {
        // Validate prefix-extension property.
        let current_fields: Vec<_> = self.schema.fields().collect();
        let proposed_fields: Vec<_> = new_schema.fields().collect();
        if proposed_fields.len() < current_fields.len() {
            return Err(crate::TantivyError::SchemaError(format!(
                "extend_schema: new schema has fewer fields ({}) than current ({}); \
                 removing fields is not supported",
                proposed_fields.len(),
                current_fields.len(),
            )));
        }
        for (idx, ((cur_field, cur_entry), (new_field, new_entry))) in
            current_fields.iter().zip(proposed_fields.iter()).enumerate()
        {
            if cur_field.field_id() != new_field.field_id() {
                return Err(crate::TantivyError::SchemaError(format!(
                    "extend_schema: field at position {idx} changed id \
                     ({} → {})",
                    cur_field.field_id(),
                    new_field.field_id(),
                )));
            }
            if cur_entry != new_entry {
                return Err(crate::TantivyError::SchemaError(format!(
                    "extend_schema: existing field {:?} at position {idx} \
                     was modified — name, type, options, or tokenizer changed",
                    cur_entry.name(),
                )));
            }
        }

        // Persist. `save_metas` is `pub(crate)` so we use it directly here.
        let mut meta = self.load_metas()?;
        meta.schema = new_schema.clone();
        crate::indexer::save_metas(&meta, &self.directory)?;
        self.directory.sync_directory()?;

        self.schema = new_schema;
        Ok(())
    }

    /// Returns the list of segments that are searchable
    pub fn searchable_segments(&self) -> crate::Result<Vec<Segment>> {
        Ok(self
            .searchable_segment_metas()?
            .into_iter()
            .map(|segment_meta| self.segment(segment_meta))
            .collect())
    }

    #[doc(hidden)]
    pub fn segment(&self, segment_meta: SegmentMeta) -> Segment {
        Segment::for_index(self.clone(), segment_meta)
    }

    /// Creates a new segment.
    pub fn new_segment(&self) -> Segment {
        let segment_meta = self
            .inventory
            .new_segment_meta(SegmentId::generate_random(), 0);
        self.segment(segment_meta)
    }

    /// Return a reference to the index directory.
    pub fn directory(&self) -> &ManagedDirectory {
        &self.directory
    }

    /// Return a mutable reference to the index directory.
    pub fn directory_mut(&mut self) -> &mut ManagedDirectory {
        &mut self.directory
    }

    /// Reads the meta.json and returns the list of
    /// `SegmentMeta` from the last commit.
    pub fn searchable_segment_metas(&self) -> crate::Result<Vec<SegmentMeta>> {
        Ok(self.load_metas()?.segments)
    }

    /// Returns the list of segment ids that are searchable.
    pub fn searchable_segment_ids(&self) -> crate::Result<Vec<SegmentId>> {
        Ok(self
            .searchable_segment_metas()?
            .iter()
            .map(SegmentMeta::id)
            .collect())
    }

    /// Returns the set of corrupted files
    pub fn validate_checksum(&self) -> crate::Result<HashSet<PathBuf>> {
        let managed_files = self.directory.list_managed_files();
        let active_segments_files: HashSet<PathBuf> = self
            .searchable_segment_metas()?
            .iter()
            .flat_map(|segment_meta| segment_meta.list_files())
            .collect();
        let active_existing_files: HashSet<&PathBuf> =
            active_segments_files.intersection(&managed_files).collect();

        let mut damaged_files = HashSet::new();
        for path in active_existing_files {
            if !self.directory.validate_checksum(path)? {
                damaged_files.insert((*path).clone());
            }
        }
        Ok(damaged_files)
    }
}

impl fmt::Debug for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Index({:?})", self.directory)
    }
}

#[cfg(test)]
mod extend_schema_tests {
    use super::*;
    use crate::doc;
    use crate::query::TermQuery;
    use crate::schema::{IndexRecordOption, Schema, STORED, STRING, TEXT};
    use crate::Term;

    fn build_schema_v1() -> Schema {
        let mut sb = Schema::builder();
        sb.add_text_field("title", TEXT | STORED);
        sb.add_f64_field("price", STORED | crate::schema::INDEXED);
        sb.build()
    }

    fn build_schema_v2_add_color() -> Schema {
        // Same fields at the same positions, plus `color` appended.
        let mut sb = Schema::builder();
        sb.add_text_field("title", TEXT | STORED);
        sb.add_f64_field("price", STORED | crate::schema::INDEXED);
        sb.add_text_field("color", STRING | STORED);
        sb.build()
    }

    #[test]
    fn extends_with_new_field_at_end() -> crate::Result<()> {
        let mut index = Index::create_in_ram(build_schema_v1());
        let title_v1 = index.schema().get_field("title")?;
        let price_v1 = index.schema().get_field("price")?;

        // Write some docs under v1.
        {
            let mut writer = index.writer_for_tests::<crate::TantivyDocument>()?;
            writer.add_document(doc!(
                title_v1 => "wireless headphones",
                price_v1 => 79.99_f64,
            ))?;
            writer.add_document(doc!(
                title_v1 => "usb cable",
                price_v1 => 12.0_f64,
            ))?;
            writer.commit()?;
        }

        // Extend the schema in place.
        let v2 = build_schema_v2_add_color();
        index.extend_schema(v2.clone())?;
        assert_eq!(index.schema().fields().count(), 3);

        // Old segments still query correctly under the extended schema.
        let title = index.schema().get_field("title")?;
        let color = index.schema().get_field("color")?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let q = TermQuery::new(
            Term::from_field_text(title, "wireless"),
            IndexRecordOption::WithFreqsAndPositions,
        );
        use crate::collector::TopDocs;
        let top = searcher.search(&q, &TopDocs::with_limit(10).order_by_score())?;
        assert_eq!(top.len(), 1, "old segment still queryable on existing field");

        // New field returns zero hits from old segment (sparse).
        let q = TermQuery::new(
            Term::from_field_text(color, "red"),
            IndexRecordOption::Basic,
        );
        let top = searcher.search(&q, &TopDocs::with_limit(10).order_by_score())?;
        assert!(top.is_empty(), "old segment has no values for the new field");

        // New writes can use the extended schema.
        {
            let mut writer = index.writer_for_tests::<crate::TantivyDocument>()?;
            writer.add_document(doc!(
                title => "bluetooth speaker",
                index.schema().get_field("price")? => 49.5_f64,
                color => "red",
            ))?;
            writer.commit()?;
        }
        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.num_docs(), 3);
        let q = TermQuery::new(
            Term::from_field_text(color, "red"),
            IndexRecordOption::Basic,
        );
        assert_eq!(searcher.search(&q, &TopDocs::with_limit(10).order_by_score())?.len(), 1);
        Ok(())
    }

    #[test]
    fn rejects_removing_a_field() {
        let mut index = Index::create_in_ram(build_schema_v2_add_color());
        // Try to "shrink" the schema by removing color.
        let mut sb = Schema::builder();
        sb.add_text_field("title", TEXT | STORED);
        sb.add_f64_field("price", STORED | crate::schema::INDEXED);
        let shrunk = sb.build();
        let err = index.extend_schema(shrunk).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("fewer fields") || msg.contains("removing"),
            "expected shrink-rejection error, got: {msg}"
        );
    }

    #[test]
    fn rejects_reordering_an_existing_field() {
        let mut index = Index::create_in_ram(build_schema_v1());
        // Swap title and price.
        let mut sb = Schema::builder();
        sb.add_f64_field("price", STORED | crate::schema::INDEXED);
        sb.add_text_field("title", TEXT | STORED);
        let reordered = sb.build();
        let err = index.extend_schema(reordered).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("modified") || msg.contains("changed") || msg.contains("position"),
            "expected reorder-rejection error, got: {msg}"
        );
    }

    #[test]
    fn index_writer_extend_schema_mid_batch() -> crate::Result<()> {
        use crate::schema::Value;
        // Build phase: write a few docs with the v1 schema, leave more
        // pending in the writer (don't commit), call extend_schema on the
        // writer, then write docs that reference the NEW field.
        let mut index = Index::create_in_ram(build_schema_v1());
        let title_v1 = index.schema().get_field("title")?;
        let price_v1 = index.schema().get_field("price")?;

        let mut writer: crate::IndexWriter = index.writer_for_tests()?;
        for i in 0..50 {
            writer.add_document(doc!(
                title_v1 => format!("old doc {i}").as_str(),
                price_v1 => (i as f64) * 1.5,
            ))?;
        }

        // No explicit commit — extend_schema must drive the flush itself.
        let new_schema = writer.extend_schema(build_schema_v2_add_color())?;

        let title = new_schema.get_field("title")?;
        let price = new_schema.get_field("price")?;
        let color = new_schema.get_field("color")?;

        // After extend_schema, the writer accepts docs referencing the new field.
        for i in 0..30 {
            writer.add_document(doc!(
                title => format!("new doc {i}").as_str(),
                price => (i as f64) * 0.5 + 100.0,
                color => if i % 2 == 0 { "red" } else { "blue" },
            ))?;
        }
        writer.commit()?;

        // The outer `index` handle still has the OLD schema clone (`Index`
        // and `IndexWriter` carry independent schema fields). Read via the
        // writer's index handle, which is the one extend_schema mutated.
        let reader = writer.index().reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.num_docs(), 80);

        // Old docs are queryable on the original fields.
        use crate::collector::TopDocs;
        use crate::query::TermQuery;
        let q = TermQuery::new(
            crate::Term::from_field_text(title, "old"),
            crate::schema::IndexRecordOption::WithFreqsAndPositions,
        );
        assert_eq!(searcher.search(&q, &TopDocs::with_limit(100).order_by_score())?.len(), 50);

        // New docs are queryable on the new field too.
        let q = TermQuery::new(
            crate::Term::from_field_text(color, "red"),
            crate::schema::IndexRecordOption::Basic,
        );
        assert_eq!(
            searcher.search(&q, &TopDocs::with_limit(100).order_by_score())?.len(),
            15,
            "new field 'color=red' matches half the new docs"
        );

        // Old docs are sparse on the new field — querying for any color
        // value never returns an old doc.
        for color_value in ["red", "blue", "green"] {
            let q = TermQuery::new(
                crate::Term::from_field_text(color, color_value),
                crate::schema::IndexRecordOption::Basic,
            );
            let top = searcher.search(&q, &TopDocs::with_limit(100).order_by_score())?;
            for (_score, addr) in top {
                let doc: crate::TantivyDocument = searcher.doc(addr)?;
                let title_text = doc
                    .get_first(title)
                    .and_then(|v| v.as_value().as_str())
                    .unwrap_or("");
                assert!(
                    !title_text.starts_with("old "),
                    "old doc should not match color={color_value}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn index_writer_extend_schema_persists_to_meta_json() -> crate::Result<()> {
        // Regression: SegmentUpdater holds its own Index clone with a
        // frozen schema field, so without SegmentUpdater::set_schema the
        // commit that follows `IndexWriter::extend_schema` writes the OLD
        // schema back to meta.json — the extension survives in memory but
        // is silently reverted on the next process restart.
        use crate::directory::RamDirectory;

        let directory = RamDirectory::create();
        let index = Index::create(directory.clone(), build_schema_v1(), Default::default())?;
        let title_v1 = index.schema().get_field("title")?;
        let price_v1 = index.schema().get_field("price")?;

        let mut writer: crate::IndexWriter = index.writer_for_tests()?;
        for i in 0..30 {
            writer.add_document(doc!(
                title_v1 => format!("old doc {i}").as_str(),
                price_v1 => (i as f64) * 1.5,
            ))?;
        }
        let new_schema = writer.extend_schema(build_schema_v2_add_color())?;
        let color = new_schema.get_field("color")?;
        for i in 0..10 {
            writer.add_document(doc!(
                new_schema.get_field("title")? => format!("new doc {i}").as_str(),
                new_schema.get_field("price")? => (i as f64) * 0.5,
                color => "red",
            ))?;
        }
        writer.commit()?;
        drop(writer);

        // Reopen the index from the directory — meta.json is the only source
        // of truth here.
        let reopened = Index::open(directory)?;
        let schema_on_disk = reopened.schema();
        assert!(
            schema_on_disk.get_field("color").is_ok(),
            "extend_schema-added field must survive in meta.json after commit"
        );
        assert_eq!(
            schema_on_disk.fields().count(),
            3,
            "reopened schema should have all 3 fields (title, price, color)"
        );

        // And queries against the new field work on the reopened index.
        let reader = reopened.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.num_docs(), 40);
        let color = schema_on_disk.get_field("color")?;
        let q = crate::query::TermQuery::new(
            crate::Term::from_field_text(color, "red"),
            crate::schema::IndexRecordOption::Basic,
        );
        use crate::collector::TopDocs;
        let top = searcher.search(&q, &TopDocs::with_limit(100).order_by_score())?;
        assert_eq!(top.len(), 10, "new field is queryable after reopen");
        Ok(())
    }

    #[test]
    fn rejects_changing_an_existing_field_options() {
        let mut index = Index::create_in_ram(build_schema_v1());
        let mut sb = Schema::builder();
        // title becomes STRING instead of TEXT — different tokenizer
        // (raw vs default) and different IndexRecordOption.
        sb.add_text_field("title", STRING | STORED);
        sb.add_f64_field("price", STORED | crate::schema::INDEXED);
        let changed = sb.build();
        let err = index.extend_schema(changed).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("modified") || msg.contains("changed") || msg.contains("options"),
            "expected option-change rejection error, got: {msg}"
        );
    }

    /// Extends the schema mid-batch, commits, adds more docs, commits a
    /// SECOND time, then runs a merge across both segments. After all of
    /// this the meta.json must still reflect the extended schema and the
    /// merged segment must still be queryable on the new field. Verifies
    /// that `SegmentUpdater::set_schema` propagates through both commits
    /// AND through `end_merge` (which also runs `save_metas`).
    #[test]
    fn index_writer_extend_schema_survives_merge() -> crate::Result<()> {
        use crate::directory::RamDirectory;

        let directory = RamDirectory::create();
        let index = Index::create(directory.clone(), build_schema_v1(), Default::default())?;
        let title_v1 = index.schema().get_field("title")?;
        let price_v1 = index.schema().get_field("price")?;

        let mut writer: crate::IndexWriter = index.writer_for_tests()?;
        // First batch under the v1 schema.
        for i in 0..20 {
            writer.add_document(doc!(
                title_v1 => format!("old {i}").as_str(),
                price_v1 => (i as f64) * 1.5,
            ))?;
        }
        let new_schema = writer.extend_schema(build_schema_v2_add_color())?;
        let new_title = new_schema.get_field("title")?;
        let new_price = new_schema.get_field("price")?;
        let color = new_schema.get_field("color")?;
        // Second batch under the extended schema — commits a separate
        // segment so the merge step has something to merge against.
        for i in 0..15 {
            writer.add_document(doc!(
                new_title => format!("new {i}").as_str(),
                new_price => (i as f64) * 0.25,
                color => if i % 3 == 0 { "red" } else { "blue" },
            ))?;
        }
        writer.commit()?;

        // Add a third batch and commit once more so we have multiple
        // segments to merge.
        for i in 0..15 {
            writer.add_document(doc!(
                new_title => format!("third {i}").as_str(),
                new_price => (i as f64) * 0.5,
                color => "green",
            ))?;
        }
        writer.commit()?;

        // Run a merge over every committed segment.
        let segment_ids = writer.index().searchable_segment_ids()?;
        assert!(segment_ids.len() >= 2, "expected at least 2 segments to merge");
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;

        // Reopen from disk — meta.json is the only source of truth now.
        let reopened = Index::open(directory)?;
        let on_disk_schema = reopened.schema();
        assert!(
            on_disk_schema.get_field("color").is_ok(),
            "the extended `color` field must persist through commit + merge"
        );
        assert_eq!(on_disk_schema.fields().count(), 3);

        let reader = reopened.reader()?;
        let searcher = reader.searcher();
        // Sanity: all 50 docs survived merge.
        assert_eq!(searcher.num_docs(), 50);

        // The new field is still queryable after the merge.
        let color = on_disk_schema.get_field("color")?;
        for (color_value, expected_count) in
            [("green", 15usize), ("blue", 10), ("red", 5)]
        {
            let q = crate::query::TermQuery::new(
                crate::Term::from_field_text(color, color_value),
                crate::schema::IndexRecordOption::Basic,
            );
            use crate::collector::TopDocs;
            let top = searcher.search(&q, &TopDocs::with_limit(100).order_by_score())?;
            assert_eq!(
                top.len(),
                expected_count,
                "color={color_value} should match {expected_count} docs after merge"
            );
        }
        Ok(())
    }
}
