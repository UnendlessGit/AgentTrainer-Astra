# Native UI rendering without privacy permissions

`python3 scripts/render_ui.py` runs the opt-in `renderActualApplicationViews` Swift test against an isolated temporary library. It renders actual product SwiftUI views through an owned off-screen `NSWindow` and `NSHostingView.cacheDisplay`. No screen capture, macOS accessibility reads, permission prompts, real input events, or model training are performed. Every window is ordered out, detached from its content, and closed after export; the fixture workspace is removed when the test process finishes.

The output lives in `.local/ui-renders/`, outside source control. `manifest.json` identifies current images and logical/pixel sizes; `render.log` retains the build/test result. Metadata, checkpoints, curves, and recording pixels are explicitly generated fixtures, not measured learning results or personal recordings. Checkpoint fixture files provide only the metadata read by these views and must never be used as real policies.

The 2026-09-11 export completed 56 base views plus 20 scrolled views. Representative full-workspace, minimum-size, light/dark, chart, interrupted-recording, and resume images were inspected directly after the fixes below. The ordinary native suite and the opt-in rendering run are separate evidence.

## Coverage

Views are rendered in Aqua and dark Aqua at both 1120 × 760 and 860 × 580 logical points. On this Mac exports use 2× backing pixels. Coverage includes:

- Full workspace sidebar/header with Library, Demonstrations, Training, and Run destinations.
- Behavioral recording-empty, generated-practice, and saved-resume configurations.
- Reinforcement configuration and progress, including experience exceeding the minimum rollout while the current episode finishes.
- Run empty/checkpoint configurations and environment guidance.
- Real Swift Charts for behavioral loss and reinforcement reward, with numeric metrics and disclosure controls.
- Recording inspector with decoded lossless frame, timeline, native event table, and a long interruption notice.

Where a real owned `NSScrollView` has overflow, the harness scrolls its clip view and exports a second `-bottom.png`. This exercises local scroll layout without sending OS input. These exports confirm that lower training/run controls and guidance remain reachable.

## Findings and corrections

Visual inspection found redundant padding inside RunView, on top of AgentWorkspace's existing padding. Removing that inner inset aligned the view with other tabs and made its primary action visible in the smaller initial viewport. Light-mode warnings used bright orange for body text on pale surfaces; warning text now uses the primary foreground while retaining an orange symbol. The recording inspector's previous 620-point minimum height clipped its header and event table in a 580-point viewport; the integrated compact dimensions preserve the full header and event rows, including the tested interruption notice.

Progress presentation is separated from the observable coordinator through small value-based content views. The application still supplies actual coordinator values; only the render harness supplies fixture values. Reinforcement charts prefer accepted-policy KL, retaining older sampled-KL metadata as a labeled fallback. Optional candidate/backtracking values remain absent when unavailable and live in a separate disclosure; a non-finite candidate is never rendered as zero. Behavioral resume hides settings and source choices that the saved training configuration owns.

## Limits

This is component rendering and visual evidence, not installed-app interactive qualification. It does not prove VoiceOver behavior, keyboard focus traversal, picker/menu operation, titlebar/toolbar integration, target capture/control, or real training success. The off-screen bitmap renderer currently paints selected native sidebar rows as an opaque black rounded rectangle, hiding their label in exports; unselected rows and native table contents render correctly. That isolated export behavior has not been established as a defect in the running application and must not be treated as final sidebar visual proof.

The ordinary native suite skips this opt-in export unless `ASTRA_UI_RENDER_DIR` is set. A passing default test run alone does not establish these screenshots. Rerun the export and inspect its current manifest/images after further layout changes.
