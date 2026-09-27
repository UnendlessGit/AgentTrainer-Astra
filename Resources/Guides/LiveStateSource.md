# Local state values

A reward definition can use provided numeric, text or Boolean values from a game or application adapter running on this Mac. These values are reward/readiness evidence, not agent inputs or keyboard shortcuts.

1. Add a Manual signal to the reward definition and enable **Connect a local state source** in desktop training.
2. Start training. Astra opens a private loopback connection and waits for current values before resetting or arming controls.
3. Use **Copy Connection Settings** for the host, port, session ID and ephemeral token. **Copy Signal IDs** identifies the fields the current reset accepts.
4. Connect your local adapter, then confirm Ready when the environment is ready. Keep sending fresh measurements within each signal's maximum age.

The example `live_signal_client.py` beside this guide uses only the Python standard library. Run `python3 live_signal_client.py --help` for a test sender, or import `LocalStateClient` and call `publish({signal_uuid: current_value})` from your adapter. The command-line test sender repeats one supplied value; replace it with real measurements for training. Python is needed only for this optional example, not for Astra itself.

The token is entered without echo by the example. It expires when the run ends. A new reset announces a new binding; the client reads that binding and the next sequence before publishing. Values are timestamped when Astra accepts them. Do not send your own timestamps or guess a sequence after a disconnect.

A missing, stale, low-confidence or explicitly unknown value is not treated as zero. A skipped update makes that reset's source unknown until a fresh binding. Required missing evidence stops learning instead of fabricating a reward. Live values never assert coverage for retrospective feedback markers; those markers are reviewed separately with controls released.
