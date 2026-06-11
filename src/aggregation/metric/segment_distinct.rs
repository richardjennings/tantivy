//! # Segment cardinality (`segment_cardinality`)
//!
//! Exact distinct-value counting with segment-local identity: the
//! result is the **sum over segments of the exact per-segment
//! distinct value count** of the field among matching documents.
//!
//! Semantics, stated precisely:
//! * On a single-segment index the value is the exact distinct
//!   count (unlike the HLL-based `cardinality`, which estimates).
//! * Across segments, a value present in S segments contributes S —
//!   a deterministic upper bound. Cross-segment dedup requires
//!   value-level identity (the `cardinality` agg's term-bytes
//!   hashing); this aggregation deliberately keeps identity
//!   segment-local and never touches the term dictionary.
//! * Under intra-segment doc-range partitioning
//!   ([`crate::collector::Collector::collect_segment_partition`])
//!   the count stays exact: fruits carry per-[`SegmentId`] sets
//!   merged by union, so a value observed by several partitions of
//!   one segment dedupes onto the same bit/element.
//!
//! Identity per column type mirrors the `cardinality` agg's salting
//! exactly: str columns use term ordinals; numeric/date/bool columns
//! use `(ColumnType as u8, value)` tuples so e.g. bool `false` and
//! i64 `0` stay distinct, and a JSON field carrying i64 and f64
//! columns counts `1` and `1.0` as two values; ip columns use the
//! u128 address.
//!
//! Not supported (v1, rejected with explicit errors): the `missing`
//! parameter, placement as a sub-aggregation under bucket
//! aggregations (consequently ordering a terms agg by this metric is
//! impossible), and sub-aggregations beneath it.

use std::fmt::Debug;

use columnar::column_values::CompactSpaceU64Accessor;
use columnar::{Column, ColumnType};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use super::cardinality::{TermOrdAccumulator, TermOrdSet};
use crate::aggregation::agg_data::AggregationsSegmentCtx;
use crate::aggregation::intermediate_agg_result::{
    IntermediateAggregationResult, IntermediateAggregationResults, IntermediateMetricResult,
};
use crate::aggregation::segment_agg_result::SegmentAggregationCollector;
use crate::aggregation::*;
use crate::index::SegmentId;
use crate::TantivyError;

/// Sparse→dense threshold for the fruit representation. Matches the
/// collection side's `PROMOTION_RATIO` in `cardinality.rs` so the
/// two layers promote consistently: dense once the distinct count
/// exceeds ~1/32 of the ord space.
const FRUIT_PROMOTION_RATIO: u64 = 32;

/// Request for a `segment_cardinality` aggregation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SegmentDistinctAggReq {
    /// The field to compute the distinct count on.
    pub field: String,
    /// Not supported by `segment_cardinality` (v1); present so the
    /// request fails with a clear error instead of being silently
    /// ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub missing: Option<Key>,
}

impl SegmentDistinctAggReq {
    /// Creates a request from a field name.
    pub fn from_field_name(field_name: String) -> Self {
        Self {
            field: field_name,
            missing: None,
        }
    }
    /// Returns the field name the aggregation is computed on.
    pub fn field_name(&self) -> &str {
        &self.field
    }
}

/// Per-(segment, column) request data for `segment_cardinality`.
pub struct SegmentDistinctAggReqData {
    /// The column accessor to access the fast field values.
    pub accessor: Column<u64>,
    /// The column_type of the field.
    pub column_type: ColumnType,
    /// The name of the aggregation.
    pub name: String,
    /// The globally-unique id of the segment being collected —
    /// the fruit's merge key. Uuid-based, so fruits from different
    /// indexes cannot collide under distributed merging.
    pub segment_id: SegmentId,
    /// The aggregation request.
    pub req: SegmentDistinctAggReq,
}

impl SegmentDistinctAggReqData {
    /// Estimate the memory consumption of this struct in bytes.
    pub fn get_memory_consumption(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Ord-set fruit representation: a sorted, duplicate-free ord list
/// while small; dense words over `[0, capacity)` once the distinct
/// count crosses `capacity / 32`. `count()` is `len` resp. popcount.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "AdaptiveOrdSetSerde", into = "AdaptiveOrdSetSerde")]
pub struct AdaptiveOrdSet {
    /// Exclusive upper bound of the segment-local ord space
    /// (column `max_value() + 1`). Identical across partitions of
    /// one segment; merging sets of different capacity is an error.
    capacity: u64,
    repr: OrdSetRepr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum OrdSetRepr {
    /// Sorted, duplicate-free ord list; its length IS the count.
    Sparse(Vec<u64>),
    /// Dense words; popcount is the count.
    Dense(Vec<u64>),
}

/// Serde shadow of [`AdaptiveOrdSet`]: validates the representation
/// invariants on deserialize so a corrupt or hand-built intermediate
/// cannot produce silently-wrong counts.
#[derive(Serialize, Deserialize)]
struct AdaptiveOrdSetSerde {
    capacity: u64,
    repr: OrdSetRepr,
}

impl From<AdaptiveOrdSet> for AdaptiveOrdSetSerde {
    fn from(value: AdaptiveOrdSet) -> Self {
        Self {
            capacity: value.capacity,
            repr: value.repr,
        }
    }
}

impl TryFrom<AdaptiveOrdSetSerde> for AdaptiveOrdSet {
    type Error = String;

    fn try_from(value: AdaptiveOrdSetSerde) -> Result<Self, String> {
        match &value.repr {
            OrdSetRepr::Sparse(ords) => {
                if !ords.windows(2).all(|w| w[0] < w[1]) {
                    return Err(
                        "AdaptiveOrdSet sparse form must be sorted and duplicate-free".to_string(),
                    );
                }
                if ords.last().is_some_and(|&last| last >= value.capacity) {
                    return Err("AdaptiveOrdSet sparse ord exceeds capacity".to_string());
                }
            }
            OrdSetRepr::Dense(words) => {
                if words.len() as u64 != value.capacity.div_ceil(64) {
                    return Err(
                        "AdaptiveOrdSet dense word length does not match capacity".to_string()
                    );
                }
            }
        }
        Ok(Self {
            capacity: value.capacity,
            repr: value.repr,
        })
    }
}

impl AdaptiveOrdSet {
    fn dense_words(capacity: u64) -> Vec<u64> {
        vec![0u64; capacity.div_ceil(64) as usize]
    }

    #[inline]
    fn set_bit(words: &mut [u64], ord: u64) {
        words[(ord / 64) as usize] |= 1u64 << (ord % 64);
    }

    /// Builds from an iterator of *unique* ords (set semantics, as
    /// produced by `TermOrdSet::iter_ords`).
    pub(crate) fn from_unique_ords(
        ords: impl Iterator<Item = u64>,
        distinct_len: u64,
        capacity: u64,
    ) -> Self {
        let repr = if distinct_len * FRUIT_PROMOTION_RATIO <= capacity {
            let mut sorted: Vec<u64> = ords.collect();
            sorted.sort_unstable();
            sorted.dedup();
            OrdSetRepr::Sparse(sorted)
        } else {
            let mut words = Self::dense_words(capacity);
            for ord in ords {
                Self::set_bit(&mut words, ord);
            }
            OrdSetRepr::Dense(words)
        };
        Self { capacity, repr }
    }

    /// Number of distinct ords in the set.
    pub(crate) fn count(&self) -> u64 {
        match &self.repr {
            OrdSetRepr::Sparse(ords) => ords.len() as u64,
            OrdSetRepr::Dense(words) => words.iter().map(|w| w.count_ones() as u64).sum(),
        }
    }

    /// Unions `other` into `self`. Both sides must describe the same
    /// segment-local ord space (equal capacity).
    pub(crate) fn merge(&mut self, other: AdaptiveOrdSet) -> crate::Result<()> {
        if self.capacity != other.capacity {
            return Err(TantivyError::InternalError(format!(
                "segment_cardinality ord-space capacity mismatch in merge: {} != {}",
                self.capacity, other.capacity
            )));
        }
        match (&mut self.repr, other.repr) {
            (OrdSetRepr::Dense(left), OrdSetRepr::Dense(right)) => {
                debug_assert_eq!(left.len(), right.len());
                for (l, r) in left.iter_mut().zip(right.iter()) {
                    *l |= *r;
                }
            }
            (OrdSetRepr::Dense(left), OrdSetRepr::Sparse(right)) => {
                for ord in right {
                    Self::set_bit(left, ord);
                }
            }
            (OrdSetRepr::Sparse(_), OrdSetRepr::Dense(mut right_words)) => {
                let OrdSetRepr::Sparse(left) = std::mem::replace(
                    &mut self.repr,
                    OrdSetRepr::Dense(Vec::new()),
                ) else {
                    unreachable!();
                };
                for ord in left {
                    Self::set_bit(&mut right_words, ord);
                }
                self.repr = OrdSetRepr::Dense(right_words);
            }
            (OrdSetRepr::Sparse(left), OrdSetRepr::Sparse(right)) => {
                // Merge-join of two sorted unique lists.
                let mut merged = Vec::with_capacity(left.len() + right.len());
                let (mut i, mut j) = (0usize, 0usize);
                while i < left.len() && j < right.len() {
                    match left[i].cmp(&right[j]) {
                        std::cmp::Ordering::Less => {
                            merged.push(left[i]);
                            i += 1;
                        }
                        std::cmp::Ordering::Greater => {
                            merged.push(right[j]);
                            j += 1;
                        }
                        std::cmp::Ordering::Equal => {
                            merged.push(left[i]);
                            i += 1;
                            j += 1;
                        }
                    }
                }
                merged.extend_from_slice(&left[i..]);
                merged.extend_from_slice(&right[j..]);
                if merged.len() as u64 * FRUIT_PROMOTION_RATIO > self.capacity {
                    let mut words = Self::dense_words(self.capacity);
                    for ord in merged {
                        Self::set_bit(&mut words, ord);
                    }
                    self.repr = OrdSetRepr::Dense(words);
                } else {
                    self.repr = OrdSetRepr::Sparse(merged);
                }
            }
        }
        Ok(())
    }
}

/// Per-segment distinct sets, one component per column-type family.
/// A field carried by several columns (JSON dynamic) merges
/// componentwise; finalization sums the component counts, matching
/// the `cardinality` agg's within-segment semantics where values of
/// different column types never alias.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SegmentDistinctEntry {
    /// Distinct term ords of the str column, if any.
    str_ords: Option<AdaptiveOrdSet>,
    /// Distinct `(column-type salt, value-bits)` tuples of
    /// numeric/date/bool columns.
    numeric: FxHashSet<(u8, u64)>,
    /// Distinct ip values (u128).
    ip: FxHashSet<u128>,
}

impl SegmentDistinctEntry {
    fn merge(&mut self, other: SegmentDistinctEntry) -> crate::Result<()> {
        match (&mut self.str_ords, other.str_ords) {
            (Some(left), Some(right)) => left.merge(right)?,
            (None, Some(right)) => self.str_ords = Some(right),
            (_, None) => {}
        }
        self.numeric.extend(other.numeric);
        self.ip.extend(other.ip);
        Ok(())
    }

    fn count(&self) -> u64 {
        self.str_ords.as_ref().map(|s| s.count()).unwrap_or(0)
            + self.numeric.len() as u64
            + self.ip.len() as u64
    }
}

/// Intermediate result of `segment_cardinality`: distinct sets keyed
/// by [`SegmentId`]. Same-key entries merge by union (doc-range
/// partitions of one segment dedupe exactly); the final value sums
/// the per-segment counts.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct IntermediateSegmentDistinct {
    per_segment: FxHashMap<SegmentId, SegmentDistinctEntry>,
}

impl IntermediateSegmentDistinct {
    pub(crate) fn from_entry(segment_id: SegmentId, entry: SegmentDistinctEntry) -> Self {
        let mut per_segment = FxHashMap::default();
        per_segment.insert(segment_id, entry);
        Self { per_segment }
    }

    pub(crate) fn merge_fruits(&mut self, other: Self) -> crate::Result<()> {
        for (segment_id, entry) in other.per_segment {
            match self.per_segment.entry(segment_id) {
                std::collections::hash_map::Entry::Occupied(mut occupied) => {
                    occupied.get_mut().merge(entry)?;
                }
                std::collections::hash_map::Entry::Vacant(vacant) => {
                    vacant.insert(entry);
                }
            }
        }
        Ok(())
    }

    /// Sum over segments of the exact per-segment distinct count.
    pub(crate) fn finalize(&self) -> f64 {
        self.per_segment.values().map(|e| e.count()).sum::<u64>() as f64
    }
}

/// Per-column collection state.
enum DistinctState {
    Str(TermOrdSet),
    Numeric(FxHashSet<(u8, u64)>),
    Ip(FxHashSet<u128>),
}

/// Segment collector for `segment_cardinality`. Top-level only:
/// build rejects nesting, and this collector defends bucket id 0.
pub(crate) struct SegmentDistinctCollector {
    /// Some until consumed by `add_intermediate_aggregation_result`.
    state: Option<DistinctState>,
    accessor_idx: usize,
    accessor: Column<u64>,
    column_type: ColumnType,
    /// Exclusive str ord-space bound (`max_value() + 1`); 0 for
    /// non-str columns.
    str_capacity: u64,
    /// Bytes already registered with the shared aggregation limits.
    reported_bytes: u64,
}

impl Debug for SegmentDistinctCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("SegmentDistinctCollector")
            .field("column_type", &self.column_type)
            .finish()
    }
}

impl SegmentDistinctCollector {
    pub(crate) fn from_req_data(accessor_idx: usize, req_data: &SegmentDistinctAggReqData) -> Self {
        let (state, str_capacity) = match req_data.column_type {
            ColumnType::Str => {
                let max_ord_inclusive = req_data.accessor.max_value();
                (
                    DistinctState::Str(TermOrdSet::new(max_ord_inclusive)),
                    max_ord_inclusive + 1,
                )
            }
            ColumnType::IpAddr => (DistinctState::Ip(FxHashSet::default()), 0),
            _ => (DistinctState::Numeric(FxHashSet::default()), 0),
        };
        Self {
            state: Some(state),
            accessor_idx,
            accessor: req_data.accessor.clone(),
            column_type: req_data.column_type,
            str_capacity,
            reported_bytes: 0,
        }
    }

    /// Rough byte estimate of the collection state, for limits
    /// registration. The shared `TermOrdSet` type is deliberately
    /// not instrumented (the `cardinality` agg's behaviour must stay
    /// untouched); this estimates from the outside.
    fn mem_estimate(&self) -> u64 {
        match &self.state {
            Some(DistinctState::Str(set)) => {
                let sparse_estimate = set.len() as u64 * 16;
                // Once promoted, the paged bitset is bounded by the
                // ord space; cap the estimate accordingly.
                sparse_estimate.min(self.str_capacity / 8 + 4096)
            }
            Some(DistinctState::Numeric(set)) => set.len() as u64 * 24,
            Some(DistinctState::Ip(set)) => set.len() as u64 * 32,
            None => 0,
        }
    }

    fn register_growth(&mut self, agg_data: &mut AggregationsSegmentCtx) -> crate::Result<()> {
        let estimate = self.mem_estimate();
        if estimate > self.reported_bytes {
            agg_data
                .context
                .limits
                .add_memory_consumed(estimate - self.reported_bytes)?;
            self.reported_bytes = estimate;
        }
        Ok(())
    }
}

impl SegmentAggregationCollector for SegmentDistinctCollector {
    fn add_intermediate_aggregation_result(
        &mut self,
        agg_data: &AggregationsSegmentCtx,
        results: &mut IntermediateAggregationResults,
        bucket_id: BucketId,
    ) -> crate::Result<()> {
        if bucket_id != 0 {
            return Err(TantivyError::InternalError(
                "segment_cardinality is a top-level aggregation; unexpected bucket id".to_string(),
            ));
        }
        let req_data = agg_data.get_segment_distinct_req_data(self.accessor_idx);
        let state = self.state.take().ok_or_else(|| {
            TantivyError::InternalError(
                "segment_cardinality state should not be finalized twice".to_string(),
            )
        })?;
        let mut entry = SegmentDistinctEntry::default();
        match state {
            DistinctState::Str(set) => {
                // Register the (transient) fruit conversion against
                // the shared budget: validates the dense-words
                // allocation, released when this guard clone drops.
                let mut limits = agg_data.context.limits.clone();
                limits.add_memory_consumed(self.str_capacity / 8 + 64)?;
                entry.str_ords = Some(AdaptiveOrdSet::from_unique_ords(
                    set.iter_ords(),
                    set.len() as u64,
                    self.str_capacity,
                ));
            }
            DistinctState::Numeric(set) => entry.numeric = set,
            DistinctState::Ip(set) => entry.ip = set,
        }
        let fruit = IntermediateSegmentDistinct::from_entry(req_data.segment_id, entry);
        results.push(
            req_data.name.clone(),
            IntermediateAggregationResult::Metric(IntermediateMetricResult::SegmentDistinct(fruit)),
        )?;
        Ok(())
    }

    fn collect(
        &mut self,
        parent_bucket_id: BucketId,
        docs: &[crate::DocId],
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        if parent_bucket_id != 0 {
            return Err(TantivyError::InternalError(
                "segment_cardinality is a top-level aggregation; unexpected bucket id".to_string(),
            ));
        }
        agg_data
            .column_block_accessor
            .fetch_block_with_missing(docs, &self.accessor, None);
        let Some(state) = &mut self.state else {
            return Err(TantivyError::InternalError(
                "collection should not happen after finalization".to_string(),
            ));
        };
        {
            let col_block_accessor = &agg_data.column_block_accessor;
            match state {
                DistinctState::Str(set) => {
                    // Promotion check on pre-block state, as the
                    // cardinality agg does.
                    set.maybe_compact();
                    set.extend_from_iter(col_block_accessor.iter_vals());
                }
                DistinctState::Numeric(set) => {
                    let salt = self.column_type as u8;
                    for val in col_block_accessor.iter_vals() {
                        set.insert((salt, val));
                    }
                }
                DistinctState::Ip(set) => {
                    let compact_space_accessor = self
                        .accessor
                        .values
                        .clone()
                        .downcast_arc::<CompactSpaceU64Accessor>()
                        .map_err(|_| {
                            TantivyError::AggregationError(
                                crate::aggregation::AggregationError::InternalError(
                                    "Type mismatch: Could not downcast to CompactSpaceU64Accessor"
                                        .to_string(),
                                ),
                            )
                        })?;
                    for val in col_block_accessor.iter_vals() {
                        let val: u128 = compact_space_accessor.compact_to_u128(val as u32);
                        set.insert(val);
                    }
                }
            }
        }
        self.register_growth(agg_data)?;
        Ok(())
    }

    fn prepare_max_bucket(
        &mut self,
        max_bucket: BucketId,
        _agg_data: &AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        if max_bucket != 0 {
            return Err(TantivyError::InternalError(
                "segment_cardinality is a top-level aggregation; unexpected bucket id".to_string(),
            ));
        }
        Ok(())
    }

    /// Never a direct child of a bucket aggregation (build rejects
    /// nesting), so ordering by it is impossible and there is no
    /// metric value to expose at segment level.
    fn compute_metric_value(
        &self,
        _bucket_id: BucketId,
        _sub_agg_name: &str,
        _sub_agg_property: &str,
        _agg_data: &AggregationsSegmentCtx,
    ) -> Option<f64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sparse(ords: &[u64], capacity: u64) -> AdaptiveOrdSet {
        AdaptiveOrdSet {
            capacity,
            repr: OrdSetRepr::Sparse(ords.to_vec()),
        }
    }

    #[test]
    fn adaptive_ord_set_merge_arms() {
        // sparse + sparse, staying sparse.
        let mut a = sparse(&[1, 5, 9], 1024);
        a.merge(sparse(&[2, 5, 700], 1024)).unwrap();
        assert_eq!(a.count(), 5);
        // sparse + sparse crossing the promotion threshold.
        let many: Vec<u64> = (0..40).map(|i| i * 2).collect();
        let more: Vec<u64> = (0..40).map(|i| i * 2 + 1).collect();
        let mut b = sparse(&many, 1024);
        b.merge(sparse(&more, 1024)).unwrap();
        assert_eq!(b.count(), 80);
        assert!(matches!(b.repr, OrdSetRepr::Dense(_)));
        // dense + sparse and sparse + dense fold identically.
        let mut c = b.clone();
        c.merge(sparse(&[999], 1024)).unwrap();
        assert_eq!(c.count(), 81);
        let mut d = sparse(&[999], 1024);
        d.merge(b.clone()).unwrap();
        assert_eq!(d.count(), 81);
        // dense + dense ORs.
        let mut e = b.clone();
        e.merge(d).unwrap();
        assert_eq!(e.count(), 81);
        // capacity mismatch errors.
        let mut f = sparse(&[1], 64);
        assert!(f.merge(sparse(&[1], 128)).is_err());
    }

    #[test]
    fn adaptive_ord_set_threshold_boundary() {
        // capacity 3200, ratio 32: <= 100 distinct stays sparse.
        let at_threshold: Vec<u64> = (0..100).collect();
        let set =
            AdaptiveOrdSet::from_unique_ords(at_threshold.iter().copied(), 100, 3200);
        assert!(matches!(set.repr, OrdSetRepr::Sparse(_)));
        let over: Vec<u64> = (0..101).collect();
        let set = AdaptiveOrdSet::from_unique_ords(over.iter().copied(), 101, 3200);
        assert!(matches!(set.repr, OrdSetRepr::Dense(_)));
        assert_eq!(set.count(), 101);
    }

    #[test]
    fn adaptive_ord_set_serde_validates() {
        let good = sparse(&[1, 2, 3], 64);
        let json = serde_json::to_string(&good).unwrap();
        let round: AdaptiveOrdSet = serde_json::from_str(&json).unwrap();
        assert_eq!(good, round);

        // Unsorted sparse form is rejected.
        let bad = json.replace("[1,2,3]", "[3,2,1]");
        assert!(serde_json::from_str::<AdaptiveOrdSet>(&bad).is_err());
        // Dense with wrong word count is rejected.
        let dense = AdaptiveOrdSet {
            capacity: 128,
            repr: OrdSetRepr::Dense(vec![0u64; 2]),
        };
        let json = serde_json::to_string(&dense).unwrap();
        let bad = json.replace("\"capacity\":128", "\"capacity\":129");
        assert!(serde_json::from_str::<AdaptiveOrdSet>(&bad).is_err());
    }
}

#[cfg(test)]
mod agg_tests {
    use std::net::IpAddr;
    use std::str::FromStr;

    use columnar::MonotonicallyMappableToU64;
    use serde_json::{json, Value};

    use crate::aggregation::agg_req::Aggregations;
    use crate::aggregation::intermediate_agg_result::IntermediateAggregationResults;
    use crate::aggregation::tests::{
        exec_request, exec_request_with_query, exec_request_with_query_and_memory_limit,
        get_test_index_from_terms,
    };
    use crate::aggregation::{AggContextParams, AggregationCollector, AggregationLimitsGuard};
    use crate::collector::Collector;
    use crate::query::{AllQuery, EnableScoring, Query, TermQuery};
    use crate::schema::{IndexRecordOption, IntoIpv6Addr, Schema, FAST, STRING};
    use crate::{Index, Term};

    fn seg_card_req(field: &str) -> Aggregations {
        serde_json::from_value(json!({
            "sc": {"segment_cardinality": {"field": field}},
        }))
        .unwrap()
    }

    fn both_aggs_req(field: &str) -> Aggregations {
        serde_json::from_value(json!({
            "sc": {"segment_cardinality": {"field": field}},
            "old": {"cardinality": {"field": field}},
        }))
        .unwrap()
    }

    // T7a: exact distinct on a single segment; the old agg agrees at
    // low cardinality.
    #[test]
    fn seg_card_str_exact() -> crate::Result<()> {
        let terms = vec![
            vec!["terma"],
            vec!["termb"],
            vec!["termc"],
            vec!["terma"],
            vec!["terma"],
            vec!["termb"],
        ];
        let index = get_test_index_from_terms(true, &terms)?;
        let res = exec_request(both_aggs_req("string_id"), &index)?;
        assert_eq!(res["sc"]["value"], 3.0);
        assert_eq!(res["old"]["value"], 3.0);
        Ok(())
    }

    // T7o: empty index and zero-match query both yield {"value": 0}.
    #[test]
    fn seg_card_empty_and_zero_match() -> crate::Result<()> {
        let index = get_test_index_from_terms(true, &[])?;
        let res = exec_request(seg_card_req("string_id"), &index)?;
        assert_eq!(res["sc"]["value"], 0.0);

        let index = get_test_index_from_terms(true, &[vec!["terma"]])?;
        let res = exec_request_with_query(
            seg_card_req("string_id"),
            &index,
            Some(("string_id", "no_such_term")),
        )?;
        assert_eq!(res["sc"]["value"], 0.0);
        Ok(())
    }

    // T7c: multi-segment upper-bound semantics, pinned — a value in
    // two segments counts twice; merged to one segment it counts
    // once.
    #[test]
    fn seg_card_multi_segment_upper_bound() -> crate::Result<()> {
        // get_test_index_from_terms: one segment per inner vec when
        // merge_segments is false.
        let terms = vec![vec!["dup"], vec!["dup"]];
        let unmerged = get_test_index_from_terms(false, &terms)?;
        let reader = unmerged.reader()?;
        assert!(reader.searcher().segment_readers().len() > 1);
        let res = exec_request(seg_card_req("string_id"), &unmerged)?;
        assert_eq!(res["sc"]["value"], 2.0, "per-segment sum counts dup twice");

        let merged = get_test_index_from_terms(true, &terms)?;
        let res = exec_request(seg_card_req("string_id"), &merged)?;
        assert_eq!(res["sc"]["value"], 1.0, "single segment is exact");
        Ok(())
    }

    // T7d: under a selective filter only matching docs contribute.
    #[test]
    fn seg_card_selective_filter() -> crate::Result<()> {
        let terms = vec![vec!["terma"], vec!["terma"], vec!["termb"]];
        let index = get_test_index_from_terms(true, &terms)?;
        let res = exec_request_with_query(
            seg_card_req("string_id"),
            &index,
            Some(("string_id", "terma")),
        )?;
        assert_eq!(res["sc"]["value"], 1.0);
        Ok(())
    }

    // T7e: every column-type arm, asserted against ground truth and
    // in parity with the old agg (same accessor representations).
    #[test]
    fn seg_card_column_types() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let f_i64 = schema_builder.add_i64_field("v_i64", FAST);
        let f_u64 = schema_builder.add_u64_field("v_u64", FAST);
        let f_f64 = schema_builder.add_f64_field("v_f64", FAST);
        let f_date = schema_builder.add_date_field("v_date", FAST);
        let f_bool = schema_builder.add_bool_field("v_bool", FAST);
        let f_ip = schema_builder.add_ip_addr_field("v_ip", FAST);
        let index = Index::create_in_ram(schema_builder.build());
        {
            let mut writer = index.writer_for_tests()?;
            let ts = |secs: i64| {
                crate::DateTime::from_utc(
                    time::OffsetDateTime::from_unix_timestamp(secs).unwrap(),
                )
            };
            let ip = |s: &str| IpAddr::from_str(s).unwrap().into_ipv6_addr();
            writer.add_document(doc!(
                f_i64 => -3i64, f_u64 => u64::MAX, f_f64 => 0.0f64,
                f_date => ts(-86_400), f_bool => true, f_ip => ip("127.0.0.1"),
            ))?;
            writer.add_document(doc!(
                f_i64 => -3i64, f_u64 => 1u64, f_f64 => -0.0f64,
                f_date => ts(-86_400), f_bool => false, f_ip => ip("10.0.0.1"),
            ))?;
            // Multi-valued numeric doc: repeated values dedupe.
            writer.add_document(doc!(
                f_i64 => 7i64, f_i64 => 7i64, f_i64 => 9i64,
                f_u64 => 1u64, f_f64 => f64::NAN, f_date => ts(0),
                f_bool => true, f_ip => ip("127.0.0.1"),
            ))?;
            writer.commit()?;
        }
        let expectations = [("v_i64", 3.0), ("v_u64", 2.0), ("v_date", 2.0), ("v_bool", 2.0)];
        for (field, want) in expectations {
            let res = exec_request(both_aggs_req(field), &index)?;
            assert_eq!(res["sc"]["value"], want, "field {field}");
            assert_eq!(res["old"]["value"], res["sc"]["value"], "parity {field}");
        }
        // f64 incl. 0.0 / -0.0 / NaN: pinned to parity with the old
        // agg (both read the same monotone u64 representation).
        let res = exec_request(both_aggs_req("v_f64"), &index)?;
        assert_eq!(res["sc"]["value"], res["old"]["value"], "f64 parity");
        // ip via the CompactSpaceU64Accessor arm.
        let res = exec_request(both_aggs_req("v_ip"), &index)?;
        assert_eq!(res["sc"]["value"], 2.0);
        assert_eq!(res["old"]["value"], 2.0);
        Ok(())
    }

    // T7e salt semantics: bool false and i64 0 under one JSON path
    // stay distinct — the (column-type, value) tuple key mirrors the
    // old agg's tuple hashing.
    #[test]
    fn seg_card_json_bool_int_salt() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let field = schema_builder.add_json_field("json", FAST);
        let index = Index::create_in_ram(schema_builder.build());
        {
            let mut writer = index.writer_for_tests()?;
            writer.add_document(doc!(field => json!({"value": false})))?;
            writer.add_document(doc!(field => json!({"value": true})))?;
            writer.add_document(doc!(field => json!({"value": i64::from_u64(0u64)})))?;
            writer.add_document(doc!(field => json!({"value": i64::from_u64(1u64)})))?;
            writer.commit()?;
        }
        let res = exec_request(both_aggs_req("json.value"), &index)?;
        assert_eq!(res["sc"]["value"], 4.0);
        assert_eq!(res["old"]["value"], 4.0);
        Ok(())
    }

    // T7f: JSON mixed str+numeric merges componentwise; numeric
    // cross-type (i64 next to f64 columns) stays in parity with the
    // old agg's salt semantics.
    #[test]
    fn seg_card_json_mixed_types() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let field = schema_builder.add_json_field("json", FAST);
        let index = Index::create_in_ram(schema_builder.build());
        {
            let mut writer = index.writer_for_tests()?;
            writer.add_document(doc!(field => json!({"value": "hello"})))?;
            writer.add_document(doc!(field => json!({"value": "world"})))?;
            writer.add_document(doc!(field => json!({"value": "hello"})))?;
            writer.add_document(doc!(field => json!({"value": 7i64})))?;
            writer.add_document(doc!(field => json!({"value": 42i64})))?;
            writer.add_document(doc!(field => json!({"value": 7i64})))?;
            writer.commit()?;
        }
        let res = exec_request(both_aggs_req("json.value"), &index)?;
        assert_eq!(res["sc"]["value"], 4.0, "hello, world, 7, 42");
        assert_eq!(res["old"]["value"], 4.0);

        // Cross numeric types: a fractional f64 forces an F64 column
        // beside the I64 column under the same path.
        let mut schema_builder = Schema::builder();
        let field = schema_builder.add_json_field("json", FAST);
        let index = Index::create_in_ram(schema_builder.build());
        {
            let mut writer = index.writer_for_tests()?;
            writer.add_document(doc!(field => json!({"value": 2i64})))?;
            writer.add_document(doc!(field => json!({"value": 1.5f64})))?;
            writer.add_document(doc!(field => json!({"value": 2i64})))?;
            writer.commit()?;
        }
        let res = exec_request(both_aggs_req("json.value"), &index)?;
        assert_eq!(res["sc"]["value"], 2.0, "i64 2 and f64 1.5");
        assert_eq!(res["old"]["value"], res["sc"]["value"], "cross-type parity");
        Ok(())
    }

    // T7g: a field absent from the segment contributes zero, no
    // error (the framework's U64 shim column).
    #[test]
    fn seg_card_field_absent() -> crate::Result<()> {
        let index = get_test_index_from_terms(true, &[vec!["terma"]])?;
        let res = exec_request(seg_card_req("no_such_field"), &index)?;
        assert_eq!(res["sc"]["value"], 0.0);
        Ok(())
    }

    // T7h: multi-valued str field.
    #[test]
    fn seg_card_multivalued_str() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let field = schema_builder.add_text_field("tags", STRING | FAST);
        let index = Index::create_in_ram(schema_builder.build());
        {
            let mut writer = index.writer_for_tests()?;
            writer.add_document(doc!(field => "a", field => "b"))?;
            writer.add_document(doc!(field => "b", field => "c"))?;
            writer.commit()?;
        }
        let res = exec_request(seg_card_req("tags"), &index)?;
        assert_eq!(res["sc"]["value"], 3.0);
        Ok(())
    }

    // T7j: nesting under a bucket aggregation is rejected.
    #[test]
    fn seg_card_nesting_rejected() -> crate::Result<()> {
        let index = get_test_index_from_terms(true, &[vec!["terma"]])?;
        let agg_req: Aggregations = serde_json::from_value(json!({
            "by_term": {
                "terms": {"field": "string_id"},
                "aggs": {
                    "sc": {"segment_cardinality": {"field": "string_id"}}
                }
            }
        }))
        .unwrap();
        let err = exec_request(agg_req, &index).unwrap_err();
        assert!(
            err.to_string().contains("top-level"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    // T7k: the missing parameter is rejected.
    #[test]
    fn seg_card_missing_rejected() -> crate::Result<()> {
        let index = get_test_index_from_terms(true, &[vec!["terma"]])?;
        let agg_req: Aggregations = serde_json::from_value(json!({
            "sc": {"segment_cardinality": {"field": "string_id", "missing": "X"}}
        }))
        .unwrap();
        let err = exec_request(agg_req, &index).unwrap_err();
        assert!(
            err.to_string().contains("missing"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    // Sub-aggregations beneath segment_cardinality are rejected.
    #[test]
    fn seg_card_subagg_rejected() -> crate::Result<()> {
        let index = get_test_index_from_terms(true, &[vec!["terma"]])?;
        let agg_req: Aggregations = serde_json::from_value(json!({
            "sc": {
                "segment_cardinality": {"field": "string_id"},
                "aggs": {"avg": {"avg": {"field": "score"}}}
            }
        }))
        .unwrap();
        let err = exec_request(agg_req, &index).unwrap_err();
        assert!(
            err.to_string().contains("sub-aggregations"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    /// Build a single-segment index where the value "common" appears
    /// in every doc (straddling every partition) beside per-chunk
    /// values v0..v7.
    fn build_straddle_index() -> Index {
        let mut schema_builder = Schema::builder();
        let field = schema_builder.add_text_field("tags", STRING | FAST);
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer = index.writer_for_tests().unwrap();
        for i in 0..512u32 {
            writer
                .add_document(doc!(
                    field => "common",
                    field => format!("v{}", i / 64),
                ))
                .unwrap();
        }
        writer.commit().unwrap();
        index
    }

    /// Drives the aggregation over K doc-range partitions and
    /// returns (final JSON, the per-partition intermediate fruits).
    /// `AggregationCollector`'s segment fruit is
    /// `crate::Result<IntermediateAggregationResults>`; the inner
    /// results are unwrapped for inspection/serde and re-wrapped
    /// with `Ok` for merging.
    fn run_partitioned(
        index: &Index,
        query: &dyn Query,
        aggs: Aggregations,
        partitions: u32,
    ) -> (Value, Vec<IntermediateAggregationResults>) {
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let seg = &searcher.segment_readers()[0];
        let collector = AggregationCollector::from_aggs(
            aggs,
            AggContextParams::new(
                AggregationLimitsGuard::new(None, None),
                index.tokenizers().clone(),
            ),
        );
        let weight = query
            .weight(EnableScoring::disabled_from_searcher(&searcher))
            .unwrap();
        let max_doc = seg.max_doc();
        let chunk = max_doc.div_ceil(partitions).max(1);
        let mut fruits: Vec<IntermediateAggregationResults> = Vec::new();
        for i in 0..partitions {
            let start = i * chunk;
            let end = ((i + 1) * chunk).min(max_doc);
            if start >= end {
                continue;
            }
            let fruit = collector
                .collect_segment_partition(weight.as_ref(), 0, seg, start..end, None)
                .unwrap();
            fruits.push(fruit.unwrap());
        }
        let final_res = collector
            .merge_fruits(fruits.iter().cloned().map(Ok).collect())
            .unwrap();
        (serde_json::to_value(&final_res).unwrap(), fruits)
    }

    // T7b + T7i: partition parity with straddling values; zero-match
    // partitions merge correctly under a selective filter.
    #[test]
    fn seg_card_partition_parity_straddling() {
        let index = build_straddle_index();
        let schema = index.schema();
        let tags = schema.get_field("tags").unwrap();

        // Match-all: "common" + v0..v7 = 9 distinct; every partition
        // observes "common".
        let (k8, _) = run_partitioned(&index, &AllQuery, seg_card_req("tags"), 8);
        let (k1, _) = run_partitioned(&index, &AllQuery, seg_card_req("tags"), 1);
        assert_eq!(k8["sc"]["value"], 9.0);
        assert_eq!(k1, k8);

        // Selective: docs holding v3 live in one chunk; the other 7
        // partitions contribute empty fruits.
        let v3 = TermQuery::new(Term::from_field_text(tags, "v3"), IndexRecordOption::Basic);
        let (k8, _) = run_partitioned(&index, &v3, seg_card_req("tags"), 8);
        assert_eq!(k8["sc"]["value"], 2.0, "common + v3");
    }

    // T7l: intermediate fruits round-trip through serde and merge to
    // the same final result.
    #[test]
    fn seg_card_intermediate_serde_roundtrip() {
        let index = build_straddle_index();
        let (direct, fruits) = run_partitioned(&index, &AllQuery, seg_card_req("tags"), 8);
        let collector = AggregationCollector::from_aggs(
            seg_card_req("tags"),
            AggContextParams::new(
                AggregationLimitsGuard::new(None, None),
                index.tokenizers().clone(),
            ),
        );
        let rounded: Vec<_> = fruits
            .into_iter()
            .map(|fruit| {
                let bytes = serde_json::to_string(&fruit).unwrap();
                let round: IntermediateAggregationResults =
                    serde_json::from_str(&bytes).unwrap();
                Ok(round)
            })
            .collect();
        let merged = collector.merge_fruits(rounded).unwrap();
        assert_eq!(serde_json::to_value(&merged).unwrap(), direct);
    }

    // T7m: collection registers growth against the shared limits.
    #[test]
    fn seg_card_limits_registration() {
        let index = build_straddle_index();
        let limits = AggregationLimitsGuard::new(None, None);
        let collector = AggregationCollector::from_aggs(
            seg_card_req("tags"),
            AggContextParams::new(limits.clone(), index.tokenizers().clone()),
        );
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        searcher.search(&AllQuery, &collector).unwrap();
        assert!(
            limits.peak_memory_for_tests() > 0,
            "collection must register memory against the shared budget"
        );
    }

    // T7p: a tiny budget fails with a clean aggregation error — no
    // panic — on the collect path.
    #[test]
    fn seg_card_limit_exceeded_clean_error() -> crate::Result<()> {
        let terms: Vec<Vec<&str>> = vec![vec!["a"], vec!["b"], vec!["c"], vec!["d"]];
        let index = get_test_index_from_terms(true, &terms)?;
        let err = exec_request_with_query_and_memory_limit(
            seg_card_req("string_id"),
            &index,
            None,
            AggregationLimitsGuard::new(Some(1), None),
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.to_lowercase().contains("memory") || msg.contains("Aborting"),
            "unexpected error: {msg}"
        );
        Ok(())
    }
}
