# AgentTrainer Astra

A local macOS application for creating agents that learn computer interaction through demonstrations and reinforcement learning.

Astra is a new implementation. Original AgentTrainer is a source of product evidence, not its architectural template. The original project is left untouched.

## Development status

Implementation is in progress. Native recording/recovery/review, behavioral and recurrent PPO training, checkpoint evaluation/resume, and local inference use one shared MLX visual/recurrent/action policy. Native training has been exercised through the assembled offline runtime; simple virtual pointing learns successfully, while the initial delayed-memory experiment remains near chance. General desktop RL, reward/reset authoring, broader learning experiments and installed capture/control qualification remain open. No completed application or release DMG is claimed. See the [implementation plan](docs/IMPLEMENTATION_PLAN.md), [verification](docs/verification/native-bundle-workflow.md) and [independent architecture](docs/architecture/0001-independent-baseline.md).

The intended release is a self-contained Apple Silicon application with a native macOS interface, MLX learning runtime, lossless recording, local inference, and an ad-hoc-signed personal-installation DMG. User recordings and checkpoints stay local.

## Build the development app

Requires Apple Silicon, Xcode with the macOS SDK, and `uv`. Build-time dependencies are downloaded into an isolated environment; the assembled compute helper includes its Python, MLX and Metal resources.

```sh
uv sync --locked --group dev
swift test
.venv/bin/python -m pytest
.venv/bin/python scripts/build_app.py
# Or build, verify, and open the native bundle:
./script/build_and_run.sh --verify
```

The development bundle is `build/AgentTrainer Astra.app`. The build verifies its ad-hoc signature and exercises the frozen compute helper with network access denied and development environment variables removed. Quit the development app before replacing it. This script does not install over another application.

The default data root is `~/Library/Application Support/AgentTrainer Astra`. `ASTRA_WORKSPACE_ROOT` overrides managed data for isolated development/testing. Recordings and checkpoints must never be committed to this repository.

Permission-free workflow checks are available through `scripts/check_native_learning.py` (assembled bundle), `scripts/render_ui.py` (actual native views in both themes), and the Python/Swift fixture suites. Live capture and input require the usual macOS privacy grants; those tests are separate from virtual practice learning.
