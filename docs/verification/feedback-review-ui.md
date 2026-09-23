# Native retrospective feedback review — September 19, 2026

The new review view/model is implemented and independently reviewed. It uses the validated retrospective source/program/artifact contract and publishes actual immutable reward revisions. It is not yet attached to a collector's pending-review source or the production desktop-learning flow; that host guard remains until source export, review ownership and learner admission are wired.

## Required owner boundary

Construct `FeedbackReviewModel` with a `VerifiedFeedbackSource`, optional loaded parent revision, local artifact directory and required `FeedbackReviewDependencies`. There is no runtime fallback. The dependencies provide:

- `validateBoundary`: verify the retained actor learning-pause token, joined physical control/guardian cleanup and the original source/policy binding. The parent must retain this ownership for the entire review lifetime; checking a Boolean once is insufficient.
- `loadObservation`: resolve the exact source/trajectory digest, episode, logical observation ID and cutoff against the verified immutable collector source. Return owned original BGRA frames, original metadata/frame IDs and their expected source-index pixel checksums. Respect the request's byte/frame budgets before allocating; completion/cancellation must join underlying I/O.
- `authoredNow`: supply the actual authoring session/clock, monotonic time and wall-time audit when the user commits a revision.

The view supports sheet/detail presentation. It disables interactive sheet dismissal until its operation completes and returns one typed outcome through `onFinish`. **An arbitrary parent/window teardown must call and await `model.cancel()` before releasing the actor pause or source loader.** `interactiveDismissDisabled` does not own an external window's close lifecycle. Parent integration also owns final source/learning admission; the view never executes a control or starts a learner.

## Review behavior

The sidebar or narrow-window previous/next controls select exact immutable intervals. Before/After selectors load the original start/endpoint observations. The model bounds loading to one in-flight request and coalesces rapid changes to the newest selection. It checks identity, metadata, cutoff, frame count, total bytes and source pixel checksums, then rechecks the owner boundary. Stale/cancelled results never replace the selected preview. Only one observation is retained; there is no synthetic preview fallback. Display uses the verified `CGImage` directly.

Counts are nonnegative integers per frozen manual-marker rule; the rule's existing signed amount determines positive/negative reward. Changing a count clears that pair's review flag. The user must load both original observations before explicitly marking a new judgment reviewed. Existing valid review/draft flags are retained on reopening. Unreviewed pairs remain unknown, including pairs with a draft count. The controls expose labels/values for accessibility, use semantic light/dark colors and support narrow scrolling layouts. No global hotkey, event suppression, screen capture or privacy prompt is added.

Save commits a new full snapshot of reviewed judgments with fresh record IDs and the actual commit authoring time. Original action/marker times, observations, policy/context inputs and earlier revisions remain unchanged. The artifact includes only reviewed-pair counts; positive counts in unreviewed pairs remain in the draft. The saved result distinguishes complete manual review from a partial revision. It is not itself permission to bypass PPO source/physics admission.

Before publication, the model durably preserves a typed draft in a separate `Drafts` folder. Drafts include source/parent binding, selection, counts and explicit review flags, never pixels or learning-ready claims. They use synchronized atomic no-overwrite publication and expected-digest loading. Cancel joins the active loader and preserves a draft without publishing a revision. If Cancel races an in-progress Save, the owned operation is joined; any already-published immutable revision is returned only as a retained/unadmitted reference. A boundary failure after publication similarly preserves the artifact and draft without reporting learning readiness. Storage failure keeps edits in memory and the window open for recovery; it never claims a durable draft exists.

## Verification

The final focused run passed **13 model tests / 18 cases**, covering exact observation loading, explicit zero versus unknown, complete and partial real publication, count changes clearing review, durable Cancel/reopen, malformed/missing/oversized/corrupt sources, single-loader coalescing, stale-result rejection, joined cancellation, source-boundary loss during opening/loading/publication, Cancel before/after immutable publication, draft tampering and storage failure/retry.

The opt-in owned rendering test also passed. It produced **16 base views plus 8 narrow scrolled views** across light/dark, 960×760 and 480×740, partial/complete review, missing source and lost-boundary states with long names. These are this test process's own `NSHostingView` pixels from generated source frames, not screen or accessibility capture. The first render pass exposed selected-row contrast and an intermittent NSImage preview snapshot; selected text contrast, direct CGImage display and bounded layout settling were corrected. The final test checks original source colors actually appear whenever a preview is present, rather than merely checking that a PNG exists. Corrected wide/dark and narrow/scrolled views were inspected.

Evidence:

- `.local/feedback-review-qualified.log`
- `.local/feedback-review-renders-qualified/manifest.json`

Reproduce with:

```sh
./script/swift.sh test --filter FeedbackReviewTests
ASTRA_FEEDBACK_RENDER_DIR="$PWD/.local/new-feedback-render-proof" ./script/swift.sh test --filter renderFeedbackReviewViews
```

An independent read-only lifecycle/admission review reported no blocker and reiterated the parent teardown/lease obligation above. Installed keyboard/VoiceOver interaction and production collector-to-review-to-PPO use remain separate qualification gates.
