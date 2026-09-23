# Native feedback workflow

One focused native integration check exercises `DesktopLearningHost` with real actor, collector and learner child processes, the production `InputExecutor` using a virtual backend, and generated capture frames. Run it with:

```sh
.venv/bin/python scripts/qualify_desktop_host.py --feedback
```

The small numerical-test recurrent policy collects complete timed episodes with an automatic reward and a manual-marker rule. At the released-control review boundary, the check opens the actual `FeedbackReviewModel`, loads the authenticated original before/after frames, acknowledges each presentation through the same model API used by the view, authors a nonzero marker and explicit reviewed zeros, and saves the immutable revision. It requires a real optimizer update, changed checkpoint weights, selected-checkpoint publication, and completed learning and pending-feedback catalog records.

A second collected batch is stopped at its review boundary. The check requires a durable draft, reopens the saved feedback through `resumeFeedback`, completes review and learning, and verifies that reopening creates no capture, actor or control process. The second checkpoint must contain a further optimizer update and changed weights.

September 23 evidence: `.local/verification/desktop-feedback-sep23-reopen1/desktop-host-report.json` and `swift-test.log`. The single native check passed in 8.809 seconds after a 2.49-second incremental build. Four episodes produced 16 reviewed intervals and 32 source-endpoint presentation acknowledgements. Both actor and collector runs exited normally; all virtual controls and capture joined. A preceding live-only run also passed in `desktop-feedback-sep23-live2`.

This is source-worker integration evidence with generated pixels and a small policy. It does not exercise a visible review window, human judgments, actual macOS input/capture or privacy permissions, a production-size policy, installed packaging, or collection of a subminimum fragment across process restarts. Those remain separate implementation/release checks. Original source/image checksums and feedback coverage validation are enforced throughout; no production validation is bypassed.
