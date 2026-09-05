# Original UI observation

After the independent baseline and source reads, the original 2.4.5 DMG was mounted read-only and its app copied into Astra's ignored local audit area. Only the copy's bundle identifier/name/signature were changed. The first launch used a distinct preferences domain, an explicit audit workspace, and denied network access. No recordings or models were imported, no recording/training/agent workflow was started, and no privacy grants were changed.

Observed through native accessibility and screenshots: Home, Record, AI Models, its Reinforcement Learning configuration, Run, Training, Library, and Settings. The supplied release bundle can display these pages on this Mac. This is UI inspection, not qualification of its learning or capture paths.

Useful observations:

- Recording presets, source choices, used-input information and reusable context are discoverable, but presets/context occupy the first screen before capture setup and the main record action can require scrolling.
- Model configuration is dense: exact pixel/color/architecture terms precede the learning workflow. The selected model is visible but configuration, training, and execution are separated across global pages.
- RL explicitly advertises the absence of a start button and requires settings across multiple pages plus a shortcut. Astra will retain visible direct entry for either learning method.
- Run exposes model vision and control scope, useful operational information, alongside lengthy implementation-oriented explanatory text. Astra will foreground active checkpoint, preview, readiness, and ownership.
- The empty training page shows many unset/zero metrics and detailed algorithm prose; useful progressive disclosure and actionable empty states can reduce initial load.
- Library search, folder totals, import/export and contextual inspection are useful. Settings mixes permissions, keybinds, reward protocol fields, appearance and storage in a long form.

The audit copy was quit. The computer-use state query after quitting restarted it; that second copy was immediately identified by its exact executable path and terminated without starting a workflow. Do not query a quit target through an API that automatically relaunches it. No Astra requirement is considered implemented by this inspection.
