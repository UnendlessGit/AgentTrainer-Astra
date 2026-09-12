# Distribution tooling and bounded packaging trial

This is packaging evidence, not a final application release. The final requested DMG remains gated by the implementation plan, live macOS workflows, ML/RL quality, accessibility and final review.

## Build and validation

`build_app.py` now keeps the compute/helper versions aligned with the native Info.plist, gathers the installed runtime notices, scans/copies required Swift compatibility libraries, removes absolute developer-machine search paths from copied native executables, adds the private Frameworks search path, and signs only after those changes. `check_bundle.py` verifies nested signatures, all three executable identities/versions/execute permissions, Apple Silicon code, pinned visual weights, Metal resources, runtime notices and contained symlinks.

Private Mach-O load edges are resolved in each executable's loader/rpath context, including dynamically loaded Python extensions. Validation selects the arm64 slice. System-library presence is checked through the public `_dyld_shared_cache_contains_path` API or filesystem; this does not load or initialize the library. Unavailable weak system dependencies remain weak, not evidence of another macOS version's compatibility. A missing private dependency or escaping relative search path fails validation. Bundle/path inputs are canonicalized, including macOS's `/var` to `/private/var` alias.

The initial trial exposed an unused absolute Xcode `swift-6.2/macosx` search path in both native executables. Swift's standard-library scanner reported no missing compatibility dylibs for those binaries. The copied binaries were normalized and re-signed before revalidation. The scanner remains part of future builds so a newly required compatibility library is embedded rather than silently assumed present.

`apple_toolchain.py` pairs the selected developer directory with its explicit `macosx` SDK without changing system selection. This Mac's unqualified SDK query returned Command Line Tools while the selected compiler came from Xcode; an explicit query selected Xcode's macOS 26.5 SDK. `script/swift.sh`, app assembly and native render/workflow scripts use that pairing. The observed host remains macOS 27.0 on an M3 Max with 36 GiB RAM; compiler target-triple output is not treated as the host OS version.

New app builds include `BuildInformation.json` with commit, dirty-source flag, hash of relevant build inputs, configuration and actual tool/runtime versions. Inputs are hashed again after assembly/verification; a changed source tree prevents replacement of the development bundle. `--require-clean` additionally rejects uncommitted source when producing a release candidate.

## DMG ownership and publication

`build_dmg.py` copies the app with `ditto`, includes an Applications link and a short installation note, builds a compressed read-only HFS+ image, verifies it, mounts privately, validates the mounted app, and validates a relocated copy. It never opens a window, asks for privacy permissions or installs over an existing app.

Cleanup discovers attached devices by the exact owned image path, including partial attach failures. An uncertain attach or failed detach retains its scratch/backing image and reports the path instead of recursively deleting a live mount. Successful cleanup verifies that neither an owned attachment nor mount remains. An exclusive persistent output lock serializes publication. The complete report and image are synchronized; the report is linked into place before the DMG, and an image-publication failure removes only the report owned by that attempt. Existing outputs are not overwritten.

Eight permanent checks in `check_packaging_faults.py` passed: failed image publication, partial attach ownership, uncertain attach cleanup, detach failure, still-attached/failed inventory, missing relative dependency, escaping loader path and non-executable entry point. These inject filesystem/command failures without mounting an image, launching code or using the GPU. Positive dependency validation precedes malformed-binary cases so an unrelated setup failure cannot satisfy a negative assertion.

## Actual trial and limits

The trial used a **copy of an earlier 0.1.0 development app**, augmented with notices and normalized native search paths. It is deliberately not the latest source/guardian/selection/collection qualification. Forty-five native code files validated. A real second trial image mounted, copied, validated and detached successfully: 203,308,155 bytes, SHA-256 `f751b6d0944b583455e4219e289b263f2472bacf133c387a6d8eaffd69724ce5`. Its private image/report/logs are under `.local/packaging-validation/`; this artifact must not be presented as the final requested DMG.

The personal-build path is ad-hoc signed, as requested. It neither requests nor asserts notarization. Mounted runtime execution, final installed TCC/HID behavior, fresh-source frozen checks and the final release DMG remain separate gates.

After all release gates are met:

```sh
uv sync --locked --group dev
./script/swift.sh test
.venv/bin/python -m pytest -W error
.venv/bin/python scripts/build_app.py --configuration release --require-clean
.venv/bin/python scripts/check_native_learning.py "build/AgentTrainer Astra.app"
.venv/bin/python scripts/check_packaging_faults.py "build/AgentTrainer Astra.app"
.venv/bin/python scripts/build_dmg.py --output dist/AgentTrainer-Astra-1.0.0-arm64.dmg
```

Update the application and Python package/runtime versions consistently before release. The commands above do not substitute for the recorded live-workflow and learning-quality evidence.
