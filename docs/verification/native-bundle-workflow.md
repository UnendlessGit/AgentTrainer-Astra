# Native coordinator and assembled runtime

On 2026-09-11, the assembled development app was built and launched through the project run entry point:

```sh
ASTRA_WORKSPACE_ROOT="/absolute/isolated/workspace" ./script/build_and_run.sh --verify
```

The script first asks an existing development process to terminate. Astra routes SIGTERM through its normal save/disarm/quit path, and the script refuses to replace a process that has not finished. It assembles the native app and control helper, freezes the Python worker, bundles the pinned visual weights/license, verifies nested ad-hoc signatures, and runs offline compute/actor plus permission-free control IPC checks before launch. The Codex Run action points to this same entry point.

`scripts/check_native_learning.py` additionally runs the real native `LearningCoordinator` against the assembled helper, rather than a Python direct-call or mock protocol. The opt-in Swift test uses an isolated library and explicit generated practice demonstrations. Three epochs with the production 34,638,639-parameter model completed six optimizer updates, published and selected the immutable trained checkpoint, then evaluated two independent validation decisions at packet NLL 8.52355. Report and test log are retained under `.local/verification/native-bundled-learning-2`. No screen, input-monitoring, Accessibility or input-posting access was used.

The first attempt to invoke that qualification encountered a concurrently edited render-test compilation failure and never ran training; its log is retained in the separate `native-bundled-learning-1` directory. The second attempt built and passed. These paths are separate so failed evidence is not overwritten by a later success.

The actual development bundle also opened an isolated empty workspace, created an agent through native UI, and reached its training view. The computer-use service then failed inside its own `Array.remove(at:)` assertion while reading accessibility state; Astra remained running and did not produce an application crash. Subsequent visual inspection uses actual SwiftUI/AppKit view renders without reading the OS screen. Those renders are complementary evidence, not proof of live keyboard/VoiceOver interaction.

Remaining gates include current final-source frozen checks after later edits, learned inference through actual macOS capture/control, installed permission attribution/revocation, recording and replay under sustained load, full desktop RL, accessibility interaction, and final DMG installation. Neither the development launch nor this simple practice-learning run is a release-completion claim.
