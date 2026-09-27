# Queued-control warm actor cost

September27,2026. One matched production-architecture comparison on AppleM3Max/36GiB, macOS27 and MLX0.32.2. The two arms share every existing parameter, with the vendored pretrained ConvNeXt and fixed random remaining weights. Schema3 adds the zero-output queue branch. This is an overhead/parity measurement, not learned-behavior evidence.

Command: `PYTHONPATH=python .venv/bin/python scripts/benchmark_queued_control.py --report .local/verification/queued-control-warm-20260927.json`.

Both arms replay the same genuinely admitted practice trajectory:1280×720 pixels,100ms decisions,200ms execution lead, full16-command historical packets, pending motion, no-op releases and original terminal evidence. The timed cuts retain three pending packets and48–64 original commands across outstanding/recent rows. The current sampled benchmark outputs are not executed. Each compiled arm receives three warm calls followed by12 measured calls; schema2 ran first. No intentional competing MLX job ran.

| Arm | Median | P95 | Maximum | Timed peak MLX bytes |
|---|---:|---:|---:|---:|
| Schema2 |59.5816ms|60.5481ms|61.0440ms|968,866,825|
| Schema3 |61.6259ms|61.9734ms|62.1281ms|940,442,951|

The observed median increment is2.0442ms/3.43%. All15 sampled packet token arrays, decoded commands and next RNG keys match exactly; maximum absolute joint-log-probability, value, entropy and recurrent-state differences are all zero. There are34,638,639 versus34,846,999 parameters. Both immutable policies remain resident and share common parameter arrays; allocator peaks are reset per arm and after warmup, while process RSS is cumulative. The lower observed schema3 allocation peak is not a claim of lower isolated process memory.

The timed region includes owned BGRA ingestion, feedback validation/continuation, shared Metal preprocessing, full-budget compiled policy/decoder, GPU completion, decoded JSON and retained feedback. It excludes ScreenCaptureKit, ring publication, frame-reference validation, subprocess IPC and parity/logging copies. Twelve warm samples in a fixed arm order do not establish tail-latency/deadline reliability. Existing live warmup/deadline checks remain required. No visual quality, packet budget or model width was reduced.

Private report SHA256: `0aecb9de89e29be8ebb265885671efaddfe45255c980731d72fee47acdb5f6cb`. The reproducible script and report retain exact seeds, model settings, input semantic digest, shared-weight digest, phases, cold/warm samples and allocation context. Installed physical timing, broader maximum-capacity conditions and queue-conditioned learning remain open.
