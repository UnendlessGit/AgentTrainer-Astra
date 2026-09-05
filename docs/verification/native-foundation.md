# Native and packaged runtime foundation

This evidence covers the initial foundation only. It does not qualify the final model, recording/input workflows, training, RL, evaluation or inference.

## Implemented and exercised

- `swift test`: 13 tests passed. Coverage includes integer clock preservation/framing faults, mixed-scale/negative-origin geometry, ordered command capabilities, geometry revisions, duplicate/stale packet rejection, lease expiry, SQLite rollback/consistent backup, exact lossless frame round-trip, read-only recovery before a partial tail, corruption detection, persistent/shared library objects, and visible corrupt-item issues.
- `.venv/bin/python -m pytest`: 8 protocol tests passed, including integer range/type checks, nonfinite JSON rejection, partial/corrupt/oversized transport, and message order.
- Native development application built and launched through its bundle. Native UI creation saved `Practice Agent` in the Astra library and selected its workspace. The default dark appearance was visually inspected. Other workspace sections are explicit empty development surfaces, not completed functionality.
- Frozen PyInstaller compute executable and nested development app helper passed `scripts/check_worker.py --offline` with PATH limited to system executables, no development environment, cwd `/tmp`, and network denied through a child-only sandbox profile. Verified hello, ping, request correlation, monotonic reply sequence, MLX Metal matrix result1240, real GRU output shape1×3×8, finite NumPy access, and orderly shutdown.
- The standalone helper and assembled native development bundle passed strict recursive code-signature verification.

## Packaging problems investigated and fixed

1. Passing the literal `-` as PyInstaller `codesign_identity` enabled hardened runtime and caused team-ID library validation to reject bundled Python. `None` selects the intended normal ad-hoc signing mode for this personal-installation build. No system security settings were changed.
2. MLX's native extension dynamically imports `mlx._reprlib_fix`; static Python analysis did not discover it. The bundle now collects runtime MLX submodules and excludes only developer extension-building/CLI entry modules. Required native libraries and Metal data are collected explicitly. The actual frozen runtime verifies this rather than relying on a successful packaging command.

PyInstaller's warning report also lists platform-conditional modules and NumPy dynamic symbols; these are not treated as proof of a runtime failure or silently counted as verified functionality. Re-run real production imports/workflows as those modules are implemented.

## Repeat

```sh
uv sync --locked --group dev
swift test
.venv/bin/python -m pytest
.venv/bin/python scripts/build_app.py
```

`build_app.py` checks the frozen helper before assembly and again inside the signed bundle. It stages replacement, retains a previous development bundle, and refuses to replace a running copy.

The final release still needs the full implementation, actual native input/capture, long-running resource/cancellation checks, real BC/RL learning, offline installed-DMG workflows, light-mode/accessibility review and final clean-build evidence from IMPLEMENTATION_PLAN.md.
