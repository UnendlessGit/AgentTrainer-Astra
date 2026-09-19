# Development machine restoration

September 19, 2026. Source, Git history, private project evidence and converted pretrained weights survived the Mac factory reset. Original AgentTrainer remains unchanged: all 57 baseline source hashes match, its HEAD matches and its Git tree is clean.

The hardware is still Apple M3 Max, 14 CPU cores and 36 GiB unified memory. The restored system reports macOS 27.0 (26A428), Xcode 27.0 (27A266a) and Swift 6.4. Earlier measurements retain their original Xcode/SDK provenance; they are not reruns on this installation.

The virtual environment's interpreter target was missing. `uv python install 3.12.13` and `uv sync --frozen --python 3.12.13` restored it without upgrading locked model dependencies. MLX 0.32.2 reports Metal available. NumPy 2.4.3, Pillow 12.3.0, PyInstaller 6.22.2 and pytest 9.1.1 load successfully. The converted ConvNeXt artifact remains present.

GitHub CLI 2.101.0 was installed under the user's local tools directory after checking the official release SHA-256. The CLI's former login did not survive; the authenticated GitHub app connection remains available. No credentials are stored in the project.

- `.local/restored-native-build.log`: native build succeeded with the paired selected compiler/SDK.
- `.local/restored-python-tests.log`: all 325 tests passed with warnings as errors in 144.41 seconds.
- `.local/restored-full-native-tests.log`: 226 reported tests passed across three test processes; 224 active and two opt-in tests skipped. The restored SwiftPM reports each test target separately.

Later integration edits receive focused regression checks and another complete build before publication. No previous macOS privacy grant, temporary `/tmp` benchmark snapshot, installed workflow or final release gate is assumed to survive restoration.

The integrated milestone then passed all 327 Python tests (`.local/integrated-python-sep19.log`) and 239 reported native tests across three target processes, with two opt-in skips (`.local/integrated-native-sep19.log`). The rebuilt signed app passed offline compute/actor checks, two collector-to-PPO checkpoint/cancellation checks, and native helper protocol checks, then launched in the isolated verification workspace (`.local/restored-frozen-bundle.log`). These are packaged-runtime checks; live capture/input, the complete desktop RL loop and release-DMG gates remain open.
