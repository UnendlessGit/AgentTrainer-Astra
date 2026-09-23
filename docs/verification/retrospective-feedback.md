# Retrospective interval feedback — September 19, 2026

The new native contract and atomic artifact writer implement the first bounded part of [ADR 0008](../architecture/0008-manual-feedback.md). A separate pure-Python reader independently validates the same artifacts without importing MLX. This contract increment did not change collector pending-review export or learner admission. A subsequent [native review UI increment](feedback-review-ui.md) implements the injected view/model and durable drafts; production source/learner wiring remains separate. Existing live `ManualRewardMarker` time semantics are unchanged.

## Source and artifact boundary

`VerifiedFeedbackSource` accepts the collector's exact eligibility-projection bytes, an externally retained expected SHA-256 and the exact frozen reward-program bytes. The expected digest must come from authenticated source/catalog evidence, not from hashing an arbitrary reviewer file and calling it trusted. The collector remains responsible for actual execution, reset, outcome, temporal-prefix and joined-cleanup eligibility. The annotation code checks this projection's declared contract; it does not substitute a Boolean or hash for that upstream audit. Native derives the bounded `manualRules` projection from the complete validated program. Python independently matches every projected manual ID/kind/signed amount against the original program under the same hash; unrelated visual/reset validation remains the authenticated native producer's responsibility.

Only `awaiting_manual_review` source projections are accepted. Each interval retains its episode/observation/packet identity, exact start/end, global sequence equal to draw index, endpoint and outcome. Valid semantic-boundary prefixes remain representable. Truncation retains the exact original same-episode/pre-reset bootstrap. Aborted/audited sources, rejected/unexecuted rows and missing or changed bootstrap cannot become reviewable here.

Annotations contain bounded integer counts for frozen manual-marker rules. Review completion explicitly covers interval/rule pairs. An unreviewed component has no count/value; reviewed-without-clicks is zero. The resolved scalar is the sum of manual components only and remains absent while any manual component is unreviewed. Other reward rules, source observations/features, actions, behavior likelihoods, recurrence, outcomes and bootstrap are untouched.

Authoring provenance is retrospective. Same-clock annotation time must follow the joined source boundary. Different clocks require distinct authoring sessions and later wall-time audit; monotonic counters from unrelated clocks are never compared. Revisions preserve original record identities/content and link to a loaded verified parent. A changed judgment uses new record identities and a new artifact rather than modifying old evidence. New records in a correction must be authored at/after its verified parent; unchanged carried records retain their original provenance.

`FeedbackArtifactStore.publish` writes and synchronizes a staging inode, links it atomically without replacing a destination, then synchronizes the directory. The artifact filename is its lower-case UUID plus `.json`; the returned reference includes its expected SHA-256. `load` checks the regular file, byte limit, no-symlink rule, expected digest, original source, parent lineage and complete re-derivation of stored reward components. It never modifies the source or an earlier revision. Larger payloads/counts are rejected by explicit bounds listed in ADR 0008.

## Verification

- Native focused suite: 11 tests / 30 cases passed, including simultaneous publication of one UUID (exactly one complete winner), no overwrite, load integrity, structural/semantic tampering, nonfinite reward rejection, symlinks, source ineligibility, exact pair/rule/count matching, duplicate identities/pairs, reviewed zero versus unknown, authoring clocks and correction lineage.
- Python focused suite: 34 tests passed with warnings as errors. Checks include immutable returned state, duplicate JSON keys, sparse oversized files, nonfinite values, rehashed semantic tampering and source eligibility/provenance.
- Direct interoperability: the native suite published a real fixture under `.local/feedback-native-fixture-projection`; Python loaded its original bytes and independently obtained manual reward 2 for the reviewed first interval, unknown for the unreviewed second interval, and unchanged source bootstrap 0.75. The process imported no MLX module.

Logs are `.local/feedback-native.log` and `.local/feedback-python.log`. These permission-free checks do not qualify the future review UI, source exporter or end-to-end PPO use of delayed manual rewards.

Run the suites with:

```sh
./script/swift.sh test --filter RetrospectiveFeedbackTests
.venv/bin/python -m pytest python/tests/test_retrospective_feedback.py -q -W error
```

Set `ASTRA_FEEDBACK_FIXTURE` to a fresh absolute directory when running the native suite to emit its standalone cross-language artifact, source/program bytes and revision reference. The environment variable is optional; normal tests use temporary directories. Reusing the fixture directory deliberately fails immutable publication. Python can reconstruct `VerifiedFeedbackSource` from those source/program bytes and the source digest, then call `load_revision` with `reference.json`. In this fixture the native test is the trusted producer; a production loader must obtain its expected source digest from the collector/catalog binding.

Independent read-only review found one cross-language validation gap in the initial draft. The explicit authenticated manual-rule projection resolves it without duplicating unrelated visual/reset validators; the reviewer reported no remaining blocker. The later-authoring requirement for new correction records was also tightened and tested before source freeze.
