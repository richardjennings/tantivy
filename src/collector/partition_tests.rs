//! Tests for `Collector::collect_segment_partition`: doc-range
//! partitions of one segment must merge to exactly what
//! `collect_segment` produces (T1/T2), partitions must equal real
//! segments (T6), scoring collectors are rejected (T3), cancellation
//! aborts (T4), and the `partition_docs` storage hint trades dense
//! for sparse terms storage without changing results (T5).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::Value;

use crate::aggregation::agg_req::Aggregations;
use crate::aggregation::{AggContextParams, AggregationCollector, AggregationLimitsGuard};
use crate::collector::{Collector, SegmentCollector};
use crate::indexer::NoMergePolicy;
use crate::query::{AllQuery, BooleanQuery, EnableScoring, Occur, PhraseQuery, Query, TermQuery};
use crate::schema::{IndexRecordOption, Schema, FAST, INDEXED, STRING, TEXT};
use crate::tokenizer::TokenizerManager;
use crate::{DocId, Index, Score, Term};

/// 2047 docs, single segment. `cat` is tie-free by construction:
/// cat_k appears 2^k times (k = 0..=10). `val` is the doc index.
/// `body` alternates "alpha beta ..." / "alpha gamma ..." so a
/// phrase query matches every even doc.
fn build_fixture() -> Index {
    let mut schema_builder = Schema::builder();
    let cat = schema_builder.add_text_field("cat", STRING | FAST);
    let val = schema_builder.add_i64_field("val", INDEXED | FAST);
    let body = schema_builder.add_text_field("body", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    let mut writer = index.writer_for_tests().unwrap();
    let mut doc_id = 0u64;
    for k in 0u32..=10 {
        for _ in 0..(1u64 << k) {
            let body_text = if doc_id % 2 == 0 {
                "alpha beta common"
            } else {
                "alpha gamma common"
            };
            writer
                .add_document(doc!(
                    cat => format!("cat_{k}"),
                    val => doc_id as i64,
                    body => body_text,
                ))
                .unwrap();
            doc_id += 1;
        }
    }
    writer.commit().unwrap();
    index
}

const AGGS_JSON: &str = r#"{
    "cats": {"terms": {"field": "cat", "size": 32}},
    "hist": {"histogram": {"field": "val", "interval": 256.0}},
    "rng": {"range": {"field": "val", "ranges": [
        {"to": 512.0}, {"from": 512.0, "to": 1024.0}, {"from": 1024.0}]}},
    "st": {"stats": {"field": "val"}}
}"#;

fn query_shapes(index: &Index) -> Vec<(&'static str, Box<dyn Query>)> {
    let schema = index.schema();
    let cat = schema.get_field("cat").unwrap();
    let body = schema.get_field("body").unwrap();
    let term_q = TermQuery::new(
        Term::from_field_text(cat, "cat_5"),
        IndexRecordOption::Basic,
    );
    // Wide Should-union: instantiates the union scorer machinery,
    // with partition boundaries landing inside union horizons.
    let union_clauses: Vec<(Occur, Box<dyn Query>)> = (3u32..=10)
        .map(|k| {
            (
                Occur::Should,
                Box::new(TermQuery::new(
                    Term::from_field_text(cat, &format!("cat_{k}")),
                    IndexRecordOption::Basic,
                )) as Box<dyn Query>,
            )
        })
        .collect();
    let union_q = BooleanQuery::new(union_clauses);
    // Must + wide Should: the shape from the seek_danger regression.
    let mut must_union_clauses: Vec<(Occur, Box<dyn Query>)> = vec![(
        Occur::Must,
        Box::new(TermQuery::new(
            Term::from_field_text(body, "alpha"),
            IndexRecordOption::Basic,
        )) as Box<dyn Query>,
    )];
    must_union_clauses.extend((2u32..=9).map(|k| {
        (
            Occur::Should,
            Box::new(TermQuery::new(
                Term::from_field_text(cat, &format!("cat_{k}")),
                IndexRecordOption::Basic,
            )) as Box<dyn Query>,
        )
    }));
    let must_union_q = BooleanQuery::new(must_union_clauses);
    let phrase_q = PhraseQuery::new(vec![
        Term::from_field_text(body, "alpha"),
        Term::from_field_text(body, "beta"),
    ]);
    vec![
        ("match_all", Box::new(AllQuery)),
        ("term", Box::new(term_q)),
        ("union", Box::new(union_q)),
        ("must_union", Box::new(must_union_q)),
        ("phrase", Box::new(phrase_q)),
    ]
}

fn fresh_params(partition_docs: Option<u32>) -> AggContextParams {
    AggContextParams {
        limits: AggregationLimitsGuard::new(None, None),
        tokenizers: TokenizerManager::default(),
        partition_docs,
    }
}

/// Runs the aggregation over the index's single segment, either via
/// legacy `collect_segment` (`partitions == None`) or as K merged
/// doc-range partitions, and returns the FINAL result as JSON.
fn agg_final(
    index: &Index,
    query: &dyn Query,
    partitions: Option<usize>,
    params: AggContextParams,
) -> crate::Result<Value> {
    let reader = index
        .reader_builder()
        .reload_policy(crate::ReloadPolicy::Manual)
        .try_into()?;
    let searcher = reader.searcher();
    assert_eq!(
        searcher.segment_readers().len(),
        1,
        "fixture must be a single segment"
    );
    let seg = &searcher.segment_readers()[0];
    let aggs: Aggregations = serde_json::from_str(AGGS_JSON).unwrap();
    let collector = AggregationCollector::from_aggs(aggs, params);
    let weight = query.weight(EnableScoring::disabled_from_searcher(&searcher))?;
    let fruits = match partitions {
        None => vec![collector.collect_segment(weight.as_ref(), 0, seg)?],
        Some(k) => {
            let max_doc = seg.max_doc();
            let chunk = max_doc.div_ceil(k as u32).max(1);
            let mut fruits = Vec::new();
            for i in 0..k as u32 {
                let start = i * chunk;
                let end = ((i + 1) * chunk).min(max_doc);
                if start >= end {
                    continue;
                }
                fruits.push(collector.collect_segment_partition(
                    weight.as_ref(),
                    0,
                    seg,
                    start..end,
                    None,
                )?);
            }
            fruits
        }
    };
    let result = collector.merge_fruits(fruits)?;
    Ok(serde_json::to_value(&result).unwrap())
}

// T1: K merged partitions == legacy collect_segment, across query
// shapes and partition counts (k=2 covers "two half-ranges merged").
#[test]
fn partition_parity_across_query_shapes() {
    let index = build_fixture();
    for (name, query) in query_shapes(&index) {
        let reference = agg_final(&index, query.as_ref(), None, fresh_params(None)).unwrap();
        for k in [2usize, 3, 8, 64] {
            let partitioned =
                agg_final(&index, query.as_ref(), Some(k), fresh_params(None)).unwrap();
            assert_eq!(reference, partitioned, "query={name} partitions={k}");
        }
    }
}

// T1 boundary cases: more partitions than docs collapses cleanly; an
// empty result set stays empty.
#[test]
fn partition_boundaries() {
    let index = build_fixture();
    let schema = index.schema();
    let cat = schema.get_field("cat").unwrap();
    // cat_0 appears exactly once: most partitions match nothing.
    let one_doc = TermQuery::new(Term::from_field_text(cat, "cat_0"), IndexRecordOption::Basic);
    let reference = agg_final(&index, &one_doc, None, fresh_params(None)).unwrap();
    let partitioned = agg_final(&index, &one_doc, Some(64), fresh_params(None)).unwrap();
    assert_eq!(reference, partitioned);

    let none = TermQuery::new(
        Term::from_field_text(cat, "no_such_cat"),
        IndexRecordOption::Basic,
    );
    let reference = agg_final(&index, &none, None, fresh_params(None)).unwrap();
    let partitioned = agg_final(&index, &none, Some(8), fresh_params(None)).unwrap();
    assert_eq!(reference, partitioned);
}

// T1 determinism: identical output across repeated partitioned runs.
#[test]
fn partition_determinism() {
    let index = build_fixture();
    let first = agg_final(&index, &AllQuery, Some(8), fresh_params(None)).unwrap();
    for _ in 0..4 {
        let again = agg_final(&index, &AllQuery, Some(8), fresh_params(None)).unwrap();
        assert_eq!(first, again);
    }
}

// T2: deletes respected — the alive-bitset arm filters identically
// on the legacy and partitioned paths.
#[test]
fn partition_parity_with_deletes() {
    let index = build_fixture();
    let schema = index.schema();
    let cat = schema.get_field("cat").unwrap();
    let mut writer = index.writer_for_tests::<crate::TantivyDocument>().unwrap();
    writer.delete_term(Term::from_field_text(cat, "cat_6"));
    writer.commit().unwrap();
    for (name, query) in query_shapes(&index) {
        let reference = agg_final(&index, query.as_ref(), None, fresh_params(None)).unwrap();
        for k in [2usize, 8] {
            let partitioned =
                agg_final(&index, query.as_ref(), Some(k), fresh_params(None)).unwrap();
            assert_eq!(reference, partitioned, "deletes query={name} partitions={k}");
        }
    }
}

/// Minimal collector whose only purpose is to declare
/// `requires_scoring() == true` (T3).
struct ScoringProbe;
struct ScoringProbeSegment;

impl Collector for ScoringProbe {
    type Fruit = usize;
    type Child = ScoringProbeSegment;

    fn for_segment(
        &self,
        _segment_local_id: u32,
        _segment: &crate::SegmentReader,
    ) -> crate::Result<Self::Child> {
        Ok(ScoringProbeSegment)
    }

    fn requires_scoring(&self) -> bool {
        true
    }

    fn merge_fruits(&self, fruits: Vec<usize>) -> crate::Result<usize> {
        Ok(fruits.into_iter().sum())
    }
}

impl SegmentCollector for ScoringProbeSegment {
    type Fruit = usize;

    fn collect(&mut self, _doc: DocId, _score: Score) {}

    fn harvest(self) -> usize {
        0
    }
}

// T3: scoring collectors are rejected with an explicit error.
#[test]
fn scoring_collector_rejected() {
    let index = build_fixture();
    let reader = index
        .reader_builder()
        .reload_policy(crate::ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    let searcher = reader.searcher();
    let seg = &searcher.segment_readers()[0];
    let weight = AllQuery
        .weight(EnableScoring::enabled_from_searcher(&searcher))
        .unwrap();
    let err = ScoringProbe
        .collect_segment_partition(weight.as_ref(), 0, seg, 0..seg.max_doc(), None)
        .unwrap_err();
    assert!(
        err.to_string().contains("does not support scoring"),
        "unexpected error: {err}"
    );
}

/// Counts collected docs; sets the shared flag after a fixed number
/// of blocks so the drive's next block check aborts (T4 mid-flight).
struct CancelAfterBlocks {
    flag: Arc<AtomicBool>,
    blocks_before_cancel: usize,
}

struct CancelAfterBlocksSegment {
    flag: Arc<AtomicBool>,
    blocks_before_cancel: usize,
    blocks_seen: usize,
    docs: usize,
}

impl Collector for CancelAfterBlocks {
    type Fruit = usize;
    type Child = CancelAfterBlocksSegment;

    fn for_segment(
        &self,
        _segment_local_id: u32,
        _segment: &crate::SegmentReader,
    ) -> crate::Result<Self::Child> {
        Ok(CancelAfterBlocksSegment {
            flag: self.flag.clone(),
            blocks_before_cancel: self.blocks_before_cancel,
            blocks_seen: 0,
            docs: 0,
        })
    }

    fn requires_scoring(&self) -> bool {
        false
    }

    fn merge_fruits(&self, fruits: Vec<usize>) -> crate::Result<usize> {
        Ok(fruits.into_iter().sum())
    }
}

impl SegmentCollector for CancelAfterBlocksSegment {
    type Fruit = usize;

    fn collect(&mut self, _doc: DocId, _score: Score) {
        self.docs += 1;
    }

    fn collect_block(&mut self, docs: &[DocId]) {
        self.docs += docs.len();
        self.blocks_seen += 1;
        if self.blocks_seen == self.blocks_before_cancel {
            self.flag.store(true, Ordering::Relaxed);
        }
    }

    fn harvest(self) -> usize {
        self.docs
    }
}

// T4: a pre-set flag aborts before any work; a mid-flight set aborts
// at the next block boundary.
#[test]
fn cancellation_pre_set_and_mid_flight() {
    let index = build_fixture();
    let reader = index
        .reader_builder()
        .reload_policy(crate::ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    let searcher = reader.searcher();
    let seg = &searcher.segment_readers()[0];
    let weight = AllQuery
        .weight(EnableScoring::disabled_from_searcher(&searcher))
        .unwrap();

    // Pre-set.
    let pre_set = AtomicBool::new(true);
    let collector = CancelAfterBlocks {
        flag: Arc::new(AtomicBool::new(false)),
        blocks_before_cancel: usize::MAX,
    };
    let err = collector
        .collect_segment_partition(weight.as_ref(), 0, seg, 0..seg.max_doc(), Some(&pre_set))
        .unwrap_err();
    assert!(
        err.to_string().contains("collection cancelled"),
        "unexpected error: {err}"
    );

    // Mid-flight: the segment collector itself trips the flag after
    // two blocks; the drive must abort with the cancellation error
    // (2047 docs = 32 blocks, so completing would need all 32).
    let flag = Arc::new(AtomicBool::new(false));
    let collector = CancelAfterBlocks {
        flag: flag.clone(),
        blocks_before_cancel: 2,
    };
    let err = collector
        .collect_segment_partition(weight.as_ref(), 0, seg, 0..seg.max_doc(), Some(&flag))
        .unwrap_err();
    assert!(
        err.to_string().contains("collection cancelled"),
        "unexpected error: {err}"
    );
    assert!(flag.load(Ordering::Relaxed));
}

/// High-cardinality fixture for the storage-hint test: every doc has
/// a unique `cat` value, so the column's max term id (4095) selects
/// the dense PagedTermMap branch unless the partition hint is set.
fn build_unique_cat_fixture() -> Index {
    let mut schema_builder = Schema::builder();
    let cat = schema_builder.add_text_field("cat", STRING | FAST);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    let mut writer = index.writer_for_tests().unwrap();
    for i in 0..4096u32 {
        writer.add_document(doc!(cat => format!("cat_{i:05}"))).unwrap();
    }
    writer.commit().unwrap();
    index
}

// T5: the partition_docs hint must not change results, and must
// lower the peak tracked memory versus K dense maps.
#[test]
fn partition_docs_hint_parity_and_memory() {
    let index = build_unique_cat_fixture();
    let schema = index.schema();
    let _cat = schema.get_field("cat").unwrap();

    let run = |partition_docs: Option<u32>| -> (Value, u64) {
        let params = fresh_params(partition_docs);
        let limits = params.limits.clone();
        let reader = index
            .reader_builder()
            .reload_policy(crate::ReloadPolicy::Manual)
            .try_into()
            .unwrap();
        let searcher = reader.searcher();
        let seg = &searcher.segment_readers()[0];
        let aggs: Aggregations =
            serde_json::from_str(r#"{"cats": {"terms": {"field": "cat", "size": 64}}}"#).unwrap();
        let collector = AggregationCollector::from_aggs(aggs, params);
        let weight = AllQuery
            .weight(EnableScoring::disabled_from_searcher(&searcher))
            .unwrap();
        let max_doc = seg.max_doc();
        let k = 8u32;
        let chunk = max_doc.div_ceil(k);
        let mut fruits = Vec::new();
        for i in 0..k {
            let start = i * chunk;
            let end = ((i + 1) * chunk).min(max_doc);
            fruits.push(
                collector
                    .collect_segment_partition(weight.as_ref(), 0, seg, start..end, None)
                    .unwrap(),
            );
        }
        let result = collector.merge_fruits(fruits).unwrap();
        (
            serde_json::to_value(&result).unwrap(),
            limits.peak_memory_for_tests(),
        )
    };

    let (dense_result, dense_peak) = run(None);
    let (sparse_result, sparse_peak) = run(Some(512));
    assert_eq!(dense_result, sparse_result);
    assert!(
        sparse_peak < dense_peak,
        "sparse peak {sparse_peak} should undercut dense peak {dense_peak}"
    );
}

// T6: K doc-range partitions of one segment are equivalent to the
// same docs written as K real segments — including the truncation
// metadata (doc_count_error_upper_bound, sum_other_doc_count) of a
// small-size terms aggregation.
#[test]
fn partitions_equal_real_segments() {
    let truncating_aggs = r#"{"cats": {"terms": {"field": "cat", "size": 3}}}"#;

    // Single-segment index.
    let single = build_fixture();

    // Same docs, same order, as 4 real segments.
    let mut schema_builder = Schema::builder();
    let cat = schema_builder.add_text_field("cat", STRING | FAST);
    let val = schema_builder.add_i64_field("val", INDEXED | FAST);
    let body = schema_builder.add_text_field("body", TEXT);
    let schema = schema_builder.build();
    let multi = Index::create_in_ram(schema);
    let mut writer = multi.writer_for_tests().unwrap();
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let mut doc_id = 0u64;
    let total_docs = 2047u64;
    let chunk = total_docs.div_ceil(4);
    let mut next_commit = chunk;
    for k in 0u32..=10 {
        for _ in 0..(1u64 << k) {
            let body_text = if doc_id % 2 == 0 {
                "alpha beta common"
            } else {
                "alpha gamma common"
            };
            writer
                .add_document(doc!(
                    cat => format!("cat_{k}"),
                    val => doc_id as i64,
                    body => body_text,
                ))
                .unwrap();
            doc_id += 1;
            if doc_id == next_commit {
                writer.commit().unwrap();
                next_commit += chunk;
            }
        }
    }
    writer.commit().unwrap();

    let run = |index: &Index, partitions: Option<u32>| -> Value {
        let reader = index
            .reader_builder()
            .reload_policy(crate::ReloadPolicy::Manual)
            .try_into()
            .unwrap();
        let searcher = reader.searcher();
        let aggs: Aggregations = serde_json::from_str(truncating_aggs).unwrap();
        let collector = AggregationCollector::from_aggs(aggs, fresh_params(None));
        let weight = AllQuery
            .weight(EnableScoring::disabled_from_searcher(&searcher))
            .unwrap();
        let mut fruits = Vec::new();
        match partitions {
            None => {
                for (ord, seg) in searcher.segment_readers().iter().enumerate() {
                    fruits.push(
                        collector
                            .collect_segment(weight.as_ref(), ord as u32, seg)
                            .unwrap(),
                    );
                }
            }
            Some(k) => {
                assert_eq!(searcher.segment_readers().len(), 1);
                let seg = &searcher.segment_readers()[0];
                let max_doc = seg.max_doc();
                let chunk = max_doc.div_ceil(k);
                for i in 0..k {
                    let start = i * chunk;
                    let end = ((i + 1) * chunk).min(max_doc);
                    fruits.push(
                        collector
                            .collect_segment_partition(weight.as_ref(), 0, seg, start..end, None)
                            .unwrap(),
                    );
                }
            }
        }
        serde_json::to_value(&collector.merge_fruits(fruits).unwrap()).unwrap()
    };

    let multi_reader = multi
        .reader_builder()
        .reload_policy(crate::ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    assert_eq!(
        multi_reader.searcher().segment_readers().len(),
        4,
        "multi fixture must hold 4 segments"
    );
    // Partition boundaries == the commit boundaries of the
    // multi-segment twin, so the two executions see identical doc
    // subsets per fruit.
    let partitioned_single = run(&single, Some(4));
    let real_segments = run(&multi, None);
    assert_eq!(partitioned_single, real_segments);
}
