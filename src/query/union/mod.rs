mod bitset_union;
mod buffered_union;
mod simple_union;

pub use bitset_union::BitSetPostingUnion;
pub use buffered_union::BufferedUnionScorer;
pub use simple_union::SimpleUnion;

#[cfg(test)]
mod tests {

    use std::collections::BTreeSet;

    use common::BitSet;

    use super::{SimpleUnion, *};
    use crate::docset::{DocSet, SeekDangerResult, TERMINATED};
    use crate::postings::tests::test_skip_against_unoptimized;
    use crate::query::score_combiner::DoNothingCombiner;
    use crate::query::union::bitset_union::BitSetPostingUnion;
    use crate::query::{BitSetDocSet, ConstScorer, Scorer, VecDocSet};
    use crate::{tests, DocId};

    fn vec_doc_set_from_docs_list(
        docs_list: &[Vec<DocId>],
    ) -> impl Iterator<Item = VecDocSet> + '_ {
        docs_list.iter().cloned().map(VecDocSet::from)
    }
    fn union_from_docs_list(docs_list: &[Vec<DocId>]) -> Box<dyn DocSet> {
        let max_doc = docs_list
            .iter()
            .flat_map(|docs| docs.iter().copied())
            .max()
            .unwrap_or(0);
        Box::new(BufferedUnionScorer::build(
            vec_doc_set_from_docs_list(docs_list)
                .map(|docset| ConstScorer::new(docset, 1.0))
                .collect::<Vec<ConstScorer<VecDocSet>>>(),
            DoNothingCombiner::default,
            max_doc,
        ))
    }

    fn posting_list_union_from_docs_list(docs_list: &[Vec<DocId>]) -> Box<dyn DocSet> {
        Box::new(BitSetPostingUnion::build(
            vec_doc_set_from_docs_list(docs_list).collect::<Vec<VecDocSet>>(),
            bitset_from_docs_list(docs_list),
        ))
    }
    fn simple_union_from_docs_list(docs_list: &[Vec<DocId>]) -> Box<dyn DocSet> {
        Box::new(SimpleUnion::build(
            vec_doc_set_from_docs_list(docs_list).collect::<Vec<VecDocSet>>(),
        ))
    }
    fn bitset_from_docs_list(docs_list: &[Vec<DocId>]) -> BitSetDocSet {
        let max_doc = docs_list
            .iter()
            .flat_map(|docs| docs.iter().copied())
            .max()
            .unwrap_or(0);
        let mut doc_bitset = BitSet::with_max_value(max_doc + 1);
        for docs in docs_list {
            for &doc in docs {
                doc_bitset.insert(doc);
            }
        }
        BitSetDocSet::from(doc_bitset)
    }
    fn aux_test_union(docs_list: &[Vec<DocId>]) {
        for constructor in [
            posting_list_union_from_docs_list,
            simple_union_from_docs_list,
            union_from_docs_list,
        ] {
            aux_test_union_with_constructor(constructor, docs_list);
        }
    }
    fn aux_test_union_with_constructor<F>(constructor: F, docs_list: &[Vec<DocId>])
    where F: Fn(&[Vec<DocId>]) -> Box<dyn DocSet> {
        let mut val_set: BTreeSet<u32> = BTreeSet::new();
        for vs in docs_list {
            for &v in vs {
                val_set.insert(v);
            }
        }
        let union_vals: Vec<u32> = val_set.into_iter().collect();
        let mut union_expected = VecDocSet::from(union_vals);
        let make_union = || constructor(docs_list);
        let mut union = make_union();
        let mut count = 0;
        while union.doc() != TERMINATED {
            assert_eq!(union_expected.doc(), union.doc());
            assert_eq!(union_expected.advance(), union.advance());
            count += 1;
        }
        assert_eq!(union_expected.advance(), TERMINATED);
        assert_eq!(count, make_union().count_including_deleted());
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_union_is_same(vecs in prop::collection::vec(
            prop::collection::vec(0u32..100, 1..10)
                .prop_map(|mut inner| {
                    inner.sort_unstable();
                    inner.dedup();
                    inner
                }),
            1..10
        ),
        seek_docids in prop::collection::vec(0u32..100, 0..10).prop_map(|mut inner| {
            inner.sort_unstable();
            inner
        })) {
            test_docid_with_skip(&vecs, &seek_docids);
        }
    }

    fn test_docid_with_skip(vecs: &[Vec<DocId>], skip_targets: &[DocId]) {
        let mut union1 = posting_list_union_from_docs_list(vecs);
        let mut union2 = simple_union_from_docs_list(vecs);
        let mut union3 = union_from_docs_list(vecs);

        // Check initial sequential advance
        while union1.doc() != TERMINATED {
            assert_eq!(union1.doc(), union2.doc());
            assert_eq!(union1.doc(), union3.doc());
            assert_eq!(union1.advance(), union2.advance());
            assert_eq!(union1.doc(), union3.advance());
        }

        // Reset and test seek functionality
        let mut union1 = posting_list_union_from_docs_list(vecs);
        let mut union2 = simple_union_from_docs_list(vecs);
        let mut union3 = union_from_docs_list(vecs);

        for &seek_docid in skip_targets {
            union1.seek(seek_docid);
            union2.seek(seek_docid);
            union3.seek(seek_docid);

            // Verify that all unions have the same document after seeking
            assert_eq!(union3.doc(), union1.doc());
            assert_eq!(union3.doc(), union2.doc());
        }
    }

    #[test]
    fn test_union() {
        aux_test_union(&[
            vec![1, 3333, 100000000u32],
            vec![1, 2, 100000000u32],
            vec![1, 2, 100000000u32],
            vec![],
        ]);
        aux_test_union(&[
            vec![1, 3333, 100000000u32],
            vec![1, 2, 100000000u32],
            vec![1, 2, 100000000u32],
            vec![],
        ]);
        aux_test_union(&[
            tests::sample_with_seed(100_000, 0.01, 1),
            tests::sample_with_seed(100_000, 0.05, 2),
            tests::sample_with_seed(100_000, 0.001, 3),
        ]);
    }

    fn test_aux_union_skip(docs_list: &[Vec<DocId>], skip_targets: Vec<DocId>) {
        for constructor in [
            posting_list_union_from_docs_list,
            simple_union_from_docs_list,
            union_from_docs_list,
        ] {
            test_aux_union_skip_with_constructor(constructor, docs_list, skip_targets.clone());
        }
    }
    fn test_aux_union_skip_with_constructor<F>(
        constructor: F,
        docs_list: &[Vec<DocId>],
        skip_targets: Vec<DocId>,
    ) where
        F: Fn(&[Vec<DocId>]) -> Box<dyn DocSet>,
    {
        let mut btree_set = BTreeSet::new();
        for docs in docs_list {
            btree_set.extend(docs.iter().cloned());
        }
        let docset_factory = || {
            let res: Box<dyn DocSet> = constructor(docs_list);
            res
        };
        let mut docset = constructor(docs_list);
        for el in btree_set {
            assert_eq!(el, docset.doc());
            docset.advance();
        }
        assert_eq!(docset.doc(), TERMINATED);
        test_skip_against_unoptimized(docset_factory, skip_targets);
    }

    #[test]
    fn test_union_skip_corner_case() {
        test_aux_union_skip(&[vec![165132, 167382], vec![25029, 25091]], vec![25029]);
    }

    // Regression: the former BufferedUnionScorer::seek_danger override
    // had no `self.doc() >= target` guard. A nested union that had
    // already advanced past the target (window re-anchored beyond it,
    // buffered matches in hand) took the out-of-horizon branch —
    // is_in_horizon's wrapping_sub turns "target behind window" into a
    // huge gap — and asked its CHILDREN, which sit beyond the buffered
    // window, fabricating a SeekLowerBound far past its own buffered
    // docs. An intersection driver then vaulted the whole window:
    // live, a Must(term) AND union-of-word-unions query lost every
    // match between the first horizon boundary (doc 4096) and the
    // children's parked positions (~doc 8200), halving facet counts.
    // Trigger: each inner union needs a postings gap right after the
    // horizon boundary so its window re-anchors past the next driver
    // targets.
    #[test]
    fn test_buffered_union_seek_danger_nested_window_guard() {
        // Inner doc lists: dense multiples up to the horizon boundary
        // (4095 = 7*585 keeps the outer window anchored via a Found),
        // then a gap so each inner union re-anchors its window beyond
        // the next driver targets. A dense driver (every doc, like a
        // high-cardinality Must term) then presents target 4096: the
        // outer goes out-of-horizon while the inners sit just ahead,
        // and the buggy override asks their children parked ~4096 docs
        // further on.
        let gappy = |step: u32, gap_from: u32, gap_to: u32| -> Vec<DocId> {
            (0u32..60_000)
                .step_by(step as usize)
                .filter(|d| *d < gap_from || *d >= gap_to)
                .collect()
        };
        let docs_a = gappy(7, 4_096, 4_403);
        let docs_b = gappy(11, 4_093, 4_301);
        let targets: Vec<DocId> = (0u32..60_000).collect();

        let in_a: std::collections::BTreeSet<DocId> = docs_a.iter().copied().collect();
        let in_b: std::collections::BTreeSet<DocId> = docs_b.iter().copied().collect();
        let expected = targets
            .iter()
            .filter(|d| in_a.contains(d) || in_b.contains(d))
            .count();

        let inner = |docs: &[DocId]| -> Box<dyn Scorer> {
            Box::new(BufferedUnionScorer::build(
                vec![ConstScorer::new(VecDocSet::from(docs.to_vec()), 1.0)],
                DoNothingCombiner::default,
                60_000,
            ))
        };
        let mut union = BufferedUnionScorer::build(
            vec![inner(&docs_a), inner(&docs_b)],
            DoNothingCombiner::default,
            60_000,
        );

        let mut found = 0usize;
        let mut idx = 0usize;
        while idx < targets.len() {
            let t = targets[idx];
            match union.seek_danger(t) {
                SeekDangerResult::Found => {
                    found += 1;
                    idx += 1;
                }
                SeekDangerResult::SeekLowerBound(lb) => {
                    if lb == TERMINATED {
                        break;
                    }
                    // Driver advances to the next target >= max(lb, t+1),
                    // mirroring Intersection::advance.
                    let next = lb.max(t + 1);
                    while idx < targets.len() && targets[idx] < next {
                        idx += 1;
                    }
                }
            }
        }
                assert_eq!(
            found, expected,
            "union lost buffered docs when seek_danger target fell behind its window"
        );
    }

    // Regression: BufferedUnionScorer's former seek_danger override
    // forwarded seek_danger into the child docsets when the target lay
    // beyond the buffered horizon. On a miss it returned with the
    // children left in the post-danger state (TermScorers block-skipped
    // without decode) while the union's own window/doc stayed stale; a
    // later hit then ran plain seeks over that inconsistent state and
    // silently skipped postings. Surfaced live as Must(term) AND
    // union-of-(per-word unions) yielding roughly half its docs under
    // scorer iteration, so facet aggregations undercounted while Count
    // stayed exact. Requires real TermScorer leaves whose secondary
    // field has postings only on docs excluded by the Must term —
    // VecDocSet children (safe default seek_danger) never trigger it.
    #[test]
    fn test_buffered_union_seek_danger_term_scorer_consistency() {
        use crate::collector::Count;
        use crate::query::{BooleanQuery, EnableScoring, Occur, Query, TermQuery};
        use crate::schema::{IndexRecordOption, Schema, STRING, TEXT};
        use crate::{doc, Index, Term};

        let mut sb = Schema::builder();
        let sch = sb.add_text_field("sch", STRING);
        let txt = sb.add_text_field("txt", TEXT);
        let fp = sb.add_text_field("fp", TEXT);
        let schema = sb.build();
        let index = Index::create_in_ram(schema);
        let mut w = index.writer_for_tests().unwrap();

        // B docs hold the query words in fp, so the fp TermScorers are
        // non-empty but never intersect the Must(sch=A) term. A docs
        // hold the words only in txt.
        let mut expected = 0usize;
        for i in 0..30_000u32 {
            if i % 3 == 0 {
                w.add_document(doc!(sch=>"B", fp=>"plc infrastructure", txt=>"baz"))
                    .unwrap();
            } else if i % 2 == 0 {
                w.add_document(doc!(sch=>"A", txt=>"foo plc", fp=>"zzz")).unwrap();
                expected += 1;
            } else if i % 5 == 0 {
                w.add_document(doc!(sch=>"A", txt=>"foo infrastructure", fp=>"zzz"))
                    .unwrap();
                expected += 1;
            } else {
                w.add_document(doc!(sch=>"A", txt=>"bar ltd", fp=>"zzz")).unwrap();
            }
        }
        w.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();

        let word = |wd: &str| -> Box<dyn Query> {
            Box::new(BooleanQuery::new(vec![
                (
                    Occur::Should,
                    Box::new(crate::query::BoostQuery::new(
                        Box::new(TermQuery::new(
                            Term::from_field_text(fp, wd),
                            IndexRecordOption::WithFreqs,
                        )),
                        3.0,
                    )) as Box<dyn Query>,
                ),
                (
                    Occur::Should,
                    Box::new(TermQuery::new(
                        Term::from_field_text(txt, wd),
                        IndexRecordOption::WithFreqs,
                    )) as Box<dyn Query>,
                ),
            ]))
        };
        let q: Box<dyn Query> = Box::new(BooleanQuery::new(vec![
            (
                Occur::Must,
                Box::new(TermQuery::new(
                    Term::from_field_text(sch, "A"),
                    IndexRecordOption::Basic,
                )) as Box<dyn Query>,
            ),
            (
                Occur::Must,
                Box::new(BooleanQuery::new(vec![
                    (Occur::Should, word("infrastructure")),
                    (Occur::Should, word("plc")),
                ])) as Box<dyn Query>,
            ),
        ]));

        let count = searcher.search(&*q, &Count).unwrap();
        assert_eq!(count, expected, "Count must match ground truth");

        let weight = q
            .weight(EnableScoring::disabled_from_searcher(&searcher))
            .unwrap();
        let mut iterated = 0usize;
        for seg in searcher.segment_readers() {
            let mut sc = weight.scorer(seg, 1.0).unwrap();
            while sc.doc() != TERMINATED {
                iterated += 1;
                sc.advance();
            }
        }
        assert_eq!(iterated, expected, "scorer iteration lost union docs");

        // Drive the union directly through the intersection protocol:
        // term docs as ascending seek_danger targets. The first target
        // beyond the union's initial buffered horizon (doc >= 4096)
        // used to flip the broken forward-to-children branch, after
        // which every later match was lost.
        let union_q: Box<dyn Query> = Box::new(BooleanQuery::new(vec![
            (Occur::Should, word("infrastructure")),
            (Occur::Should, word("plc")),
        ]));
        let term_q: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(sch, "A"),
            IndexRecordOption::Basic,
        ));
        let mut protocol_found = 0usize;
        for seg in searcher.segment_readers() {
            let scoring = EnableScoring::disabled_from_searcher(&searcher);
            let mut term_sc = term_q.weight(scoring).unwrap().scorer(seg, 1.0).unwrap();
            let mut union_sc = union_q
                .weight(EnableScoring::disabled_from_searcher(&searcher))
                .unwrap()
                .scorer(seg, 1.0)
                .unwrap();
            let mut d = term_sc.doc();
            while d != TERMINATED {
                match union_sc.seek_danger(d) {
                    SeekDangerResult::Found => {
                        protocol_found += 1;
                        d = term_sc.advance();
                    }
                    SeekDangerResult::SeekLowerBound(lb) => {
                        if lb == TERMINATED {
                            break;
                        }
                        d = term_sc.seek(lb.max(d + 1));
                    }
                }
            }
        }
        assert_eq!(
            protocol_found, expected,
            "seek_danger protocol lost union docs past the buffered horizon"
        );
    }

    #[test]
    fn test_union_skip_corner_case2() {
        test_aux_union_skip(
            &[vec![1u32, 1u32 + 100], vec![2u32, 1000u32, 10_000u32]],
            vec![0u32, 1u32, 2u32, 3u32, 1u32 + 100, 2u32 + 100],
        );
    }

    #[test]
    fn test_union_skip_corner_case3() {
        let mut docset = posting_list_union_from_docs_list(&[vec![0u32, 5u32], vec![1u32, 4u32]]);
        assert_eq!(docset.doc(), 0u32);
        assert_eq!(docset.seek(0u32), 0u32);
        assert_eq!(docset.seek(0u32), 0u32);
        assert_eq!(docset.doc(), 0u32)
    }

    #[test]
    fn test_union_skip_random() {
        test_aux_union_skip(
            &[
                vec![1, 2, 3, 7],
                vec![1, 3, 9, 10000],
                vec![1, 3, 8, 9, 100],
            ],
            vec![1, 2, 3, 5, 6, 7, 8, 100],
        );
        test_aux_union_skip(
            &[
                tests::sample_with_seed(100_000, 0.001, 1),
                tests::sample_with_seed(100_000, 0.002, 2),
                tests::sample_with_seed(100_000, 0.005, 3),
            ],
            tests::sample_with_seed(100_000, 0.01, 4),
        );
    }

    #[test]
    fn test_union_skip_specific() {
        test_aux_union_skip(
            &[
                vec![1, 2, 3, 7],
                vec![1, 3, 9, 10000],
                vec![1, 3, 8, 9, 100],
            ],
            vec![1, 2, 3, 7, 8, 9, 99, 100, 101, 500, 20000],
        );
    }

    #[test]
    fn test_buffered_union_seek_into_danger_zone_terminated() {
        let scorer1 = ConstScorer::new(VecDocSet::from(vec![1, 2]), 1.0);
        let scorer2 = ConstScorer::new(VecDocSet::from(vec![2, 3]), 1.0);

        let mut union_scorer =
            BufferedUnionScorer::build(vec![scorer1, scorer2], DoNothingCombiner::default, 100);

        // Advance to end
        while union_scorer.doc() != TERMINATED {
            union_scorer.advance();
        }

        assert_eq!(union_scorer.doc(), TERMINATED);

        assert_eq!(
            union_scorer.seek_danger(TERMINATED),
            SeekDangerResult::SeekLowerBound(TERMINATED)
        );
    }
}

#[cfg(all(test, feature = "unstable"))]
mod bench {

    use test::Bencher;

    use crate::query::score_combiner::DoNothingCombiner;
    use crate::query::{BufferedUnionScorer, ConstScorer, VecDocSet};
    use crate::{tests, DocId, DocSet, TERMINATED};

    #[bench]
    fn bench_union_3_high(bench: &mut Bencher) {
        let union_docset: Vec<Vec<DocId>> = vec![
            tests::sample_with_seed(100_000, 0.1, 0),
            tests::sample_with_seed(100_000, 0.2, 1),
        ];
        bench.iter(|| {
            let mut v = BufferedUnionScorer::build(
                union_docset
                    .iter()
                    .map(|doc_ids| VecDocSet::from(doc_ids.clone()))
                    .map(|docset| ConstScorer::new(docset, 1.0))
                    .collect::<Vec<_>>(),
                DoNothingCombiner::default,
                100_000,
            );
            while v.doc() != TERMINATED {
                v.advance();
            }
        });
    }
    #[bench]
    fn bench_union_3_low(bench: &mut Bencher) {
        let union_docset: Vec<Vec<DocId>> = vec![
            tests::sample_with_seed(100_000, 0.01, 0),
            tests::sample_with_seed(100_000, 0.05, 1),
            tests::sample_with_seed(100_000, 0.001, 2),
        ];
        bench.iter(|| {
            let mut v = BufferedUnionScorer::build(
                union_docset
                    .iter()
                    .map(|doc_ids| VecDocSet::from(doc_ids.clone()))
                    .map(|docset| ConstScorer::new(docset, 1.0))
                    .collect::<Vec<_>>(),
                DoNothingCombiner::default,
                100_000,
            );
            while v.doc() != TERMINATED {
                v.advance();
            }
        });
    }
}
