# EchoSeg UI Functional Specification

Quick user-facing reference: [UI_QUICK_REFERENCE.md](/home/mahobley/Code/echo-seg-user-pipeline/UI_QUICK_REFERENCE.md)

## 1. Purpose and scope

This document specifies the behavior of the existing desktop UI in `echo_seg_app/` closely enough that a new GUI could be built against the same Python backend and preserve the same functionality.

This is a functional spec, not a visual design spec. It focuses on:

- all visible controls
- default values
- enable/disable behavior
- validation rules
- saved settings
- run-time state changes
- progress/logging behavior
- output file behavior

The current implementation is a single-window PySide6 desktop app titled:

- `TaRDIS — ARIS Echogram Segmentation`

The window opens at approximately:

- `720 x 640`

## 2. Primary workflow

The GUI supports this workflow:

1. Choose a model checkpoint.
2. Choose one or more `.aris` or `.ddf` sonar files, or choose a directory containing them.
3. Optionally choose an output directory and processing options.
4. Review the batch preview showing how many files will run vs be skipped.
5. Run the pipeline.
6. Watch status text, progress bar, ETA, and live logs.
7. Open the output folder or log file after/during execution.

## 3. Top-level window layout

The main window contains these sections in this order:

1. `Inputs` group box
2. `Model` group box
3. `Options` group box
4. `Advanced` header row plus collapsible content
5. Action buttons row
6. Single-line status label
7. Batch plan / progress panel
8. Read-only log output pane

## 4. Inputs section

### 4.1 Checkpoints

Controls:

- `Checkpoints:` combo box
- `Use other checkpoint` checkbox
- `Other checkpoint:` line edit
- `Browse…` button

Behavior:

- The combo box is populated from `weights/*.pt`.
- In a frozen build, checkpoints are read from a `weights/` folder beside the executable if that folder exists.
- Otherwise checkpoints are read from the repo `weights/` folder.
- The combo box always includes a first item `(none)` with an empty value.
- If at least one bundled checkpoint exists, the default selection is the first bundled `.pt` file.
- The combo box tooltip is: `Weights in dist/EchoSeg/weights/ or repo weights/`.

`Use other checkpoint` behavior:

- Default: unchecked.
- When unchecked:
  - the combo box is enabled
  - the custom path line edit and browse button are disabled
  - the active checkpoint path comes from the combo box selection
- When checked:
  - the combo box is disabled
  - the custom path line edit and browse button are enabled
  - if the custom path field is blank, it is prefilled from the current combo box selection if available

Custom checkpoint browse behavior:

- Opens a file picker.
- File filter: `PyTorch weights (*.pt);;All files (*)`

Validation:

- The resolved checkpoint path must exist and be a file.
- If not, run validation fails with `Checkpoint not found: <path>`.

Persistence:

- The selected checkpoint path is saved across launches.
- Whether `Use other checkpoint` is enabled is saved across launches.

### 4.2 ARIS input selection

Controls:

- `ARIS input:` read-only line edit used as a summary field
- `Add files…` button
- `Add directory…` button
- `Clear` button

Accepted file types:

- `.aris`
- `.ddf`

#### Add files behavior

- Opens a multi-file picker.
- File filter: `Sonar files (*.aris *.ddf);;All files (*)`
- Selected files are appended to the current file selection, not replaced.
- Duplicate files are removed by resolved path.
- The stored list is sorted alphabetically by filename, case-insensitive.
- Choosing files clears directory mode by setting the source directory to `None`.
- Choosing files disables the directory-selection options group.

Summary field text after file selection:

- no files: blank field with placeholder `No files selected`
- one file: full path of the selected file
- multiple files: `<N> files selected`

#### Add directory behavior

- Opens a directory picker.
- The selected directory becomes the active source directory.
- Directory mode options become enabled.
- The file list is regenerated from the directory based on the active directory mode.
- The batch preview is refreshed immediately.

Directory discovery rules:

- Only the immediate contents of the chosen directory are scanned.
- Scanning is not recursive.
- Only files with suffix `.aris` or `.ddf` are included.
- Files are sorted alphabetically by filename, case-insensitive.

Directory summary field text:

- `<directory path> — <N> file(s), all`
- `<directory path> — <N> file(s), first <limit>`

#### Directory selection options

Controls inside `Directory selection` group:

- radio button `All files in directory (A-Z)`
- radio button `First N files in directory (A-Z)`
- spin box for `N`

Behavior:

- The whole group is disabled unless a directory has been chosen.
- Default mode: `All files in directory (A-Z)`.
- The `N` spin box is enabled only when `First N files in directory (A-Z)` is selected.
- `N` range: `1` to `1,000,000`
- `N` default: `10`
- In limited mode, the selected file list is the first `N` files from the alphabetically sorted directory listing.

Errors:

- If the chosen directory is invalid or unreadable, show warning dialog `Directory error`.
- If no `.aris` or `.ddf` files are found, show warning dialog `No files found`.

#### Clear behavior

- Clears the selected file list.
- Clears the active source directory.
- Disables the directory-selection options group.
- Clears the summary field.
- Refreshes the batch preview.

### 4.3 Skip already processed

Control:

- checkbox `Skip already processed`

Default:

- checked

Tooltip:

- `Skip files whose enabled export outputs already exist in the output directory`

Behavior:

- Changing it immediately refreshes the batch preview.
- During batch planning, a file is considered already processed only if all currently enabled output files already exist.

Per-file skip detection checks these exact expected files:

- PNG: `<parent-folder>__<stem>_<start_frame>_<end_frame>_predictions.png`
- CSV: `<stem>_<start_frame>_<end_frame>_predictions.csv`
- FC: `FCe_<stem>_ID_.txt`
- Echotastic: `<stem>.aris.txt`

Output directory used for skip detection:

- if output directory is set: that directory
- otherwise: the input file's parent directory

### 4.4 Output directory

Controls:

- `Output directory:` line edit
- `Browse…` button

Behavior:

- The line edit is manually editable.
- The browse dialog only selects an existing directory, but the user may type a new path manually.

Browse behavior:

- Opens a directory picker for an existing directory.
- The picker starts from the current text value if present, else the home directory.

Defaulting behavior:

- If the field is blank when inputs are chosen, the GUI may auto-fill it:
  - after choosing a directory input: `<selected directory>/outputs`
  - after choosing one file: `<file parent>/outputs`
  - after choosing multiple files: `<first selected file parent>/outputs`
- Auto-fill only happens if the output field is currently blank.

Validation:

- For multiple input files, an output directory is required.
- For a single input file, the output directory may be left blank.
- If left blank for a single-file run, outputs are written to the input file's parent directory.
- If a non-blank output directory is supplied, it is created automatically if needed.

## 5. Options section

The Options section groups runtime controls and export toggles. Layout notes (as implemented in the current UI):

- `Upstream:` radio buttons appear first.
- `Analyse all frames` and the `Start` / `End` controls share the next row.
- `Export FC`, `Export CSV`, and `Export Echotastic` appear together on one row.
- `Export PNG`, `Include input echogram`, and `Include summary bar` appear together on the next row.

### 5.1 Upstream direction

Controls:

- radio button `Left`
- radio button `Right`

Default:

- `Left`

Label:

- `Upstream:`

Tooltips:

- Left: `pos → Down, neg → Up in FC Dir column`
- Right: `pos → Up, neg → Down in FC Dir column`

Behavior and mapping rules are unchanged from prior spec: this affects FC and Echotastic exports but not PNG/CSV rendering.

### 5.2 Frames and "Analyse all frames"

Controls:

- checkbox `Analyse all frames`
- spin box `Start` (numeric)
- spin box `End` (numeric, displays `end of file` at `-1`)

Defaults:

- `Analyse all frames`: checked
- `Start`: 0
- `End`: -1

Behavior:

- When `Analyse all frames` is checked:
  - `Start` and `End` are disabled (greyed out) and their effective values are `0` and `-1` respectively.
  - The UI still includes these values in generated output filenames (i.e., `0` and `-1` are used).
- When `Analyse all frames` is unchecked:
  - `Start` and `End` become editable and are passed to the backend as entered.
- Toggling the checkbox refreshes the batch preview.

Validation:

- `Start` range: `0` to `10,000,000`.
- `End` range: `-1` to `10,000,000`.

### 5.3 Export toggles and PNG overlay

Controls:

- checkbox `Export FC` (single-line grouped)
- checkbox `Export CSV` (single-line grouped)
- checkbox `Export Echotastic` (single-line grouped)
- checkbox `Export PNG` (separate line)
- checkbox `Include input echogram` (enabled only when `Export PNG` is checked)
- checkbox `Include summary bar` (enabled only when `Export PNG` is checked)

Defaults:

- `Export PNG`: unchecked
- `Export CSV`: checked
- `Export FC`: checked
- `Export Echotastic`: unchecked
- `Include input echogram`: checked
- `Include summary bar`: checked

Behavior:

- `Export FC`, `Export CSV`, and `Export Echotastic` are presented together for quick toggling.
- `Export PNG` is presented separately with the `Include input echogram` and `Include summary bar` options.
- When `Include input echogram` is checked, the PNG includes the input echogram in the exported visualization.
- When `Include summary bar` is checked, the PNG includes the summary/title bar above the visualization.
- `Include input echogram` and `Include summary bar` are disabled (greyed out) whenever `Export PNG` is unchecked.
- Changes to these toggles refresh the batch preview immediately.

### 5.4 Start/End filename semantics

The `Start` and `End` values are included verbatim in output filenames. `End=-1` remains the canonical representation for "end of file" used in output names.

## 6. Advanced section

The section is hidden by default and controlled by a toggle button labeled `Advanced`.

The same header row also contains a `Reset all` button aligned to the right.

Toggle behavior:

- collapsed by default
- checkable
- shows a right arrow when collapsed
- shows a down arrow when expanded

`Reset all` behavior:

- resets all Advanced controls to their defaults
- does not affect inputs, outputs, checkpoint selection, or non-Advanced options

### 6.1 Filter submasks

Control:

- checkbox `Filter submasks`

Default:

- checked

Behavior:

- Passed to inference/export code as `filter_submasks`.

### 6.2 Confidence

Control:

- spin box labeled `Confidence:`

Default:

- `0.10`

Behavior:

- Passed to inference/export code as `conf`

### 6.3 IoU

Control:

- spin box labeled `IoU:`

Default:

- `0.50`

Behavior:

- Passed to inference/export code as `iou`

### 6.4 Inference FPS

Controls:

- checkbox `Native FPS`
- double spin box in the `Inference FPS:` row

Default:

- `Native FPS` checked
- manual FPS value initialized to `1.0`

Behavior:

- when `Native FPS` is checked:
  - the spin box is disabled
  - backend value is `-1.0`
- when `Native FPS` is unchecked:
  - the spin box is enabled
  - allowed range is `1.0` to `120.0`
  - step size is `1.0`
  - displayed precision is 1 decimal place
  - backend value is the spin box value

Persistence:

- restored across launches
- backward compatible with older saved values where `infer_fps=-1` meant native mode

### 6.5 Inference bins

Controls:

- checkbox `Native inference bins`
- combo box in the `Inference bins:` row

Visible choices when manual mode is active:

- `640`
- `1280`
- `1920`
- `2560`

Default:

- `Native inference bins` checked
- manual combo box default selection is `1280`

Behavior:

- when `Native inference bins` is checked:
  - the combo box is disabled
  - backend value is `-1`
- when `Native inference bins` is unchecked:
  - the combo box is enabled
  - backend value is the selected preset

Persistence:

- restored across launches
- if no prior value exists, native mode is used

### 6.6 Device

Control:

- combo box labeled `Device:`

Population rules:

- Always includes `cpu`.
- The app startup code sets `CUDA_VISIBLE_DEVICES` to an empty string unless the environment already defines it.
- Therefore, in a normal launch the dropdown is typically CPU-only even on GPU-capable machines.
- If CUDA is visible to the process and `torch.cuda.is_available()` is true, append:
  - `cuda:0`
  - `cuda:1`
  - etc. through `torch.cuda.device_count() - 1`

Default:

- first available entry, normally `cpu`

Persistence:

- The previously selected device is restored if still available.

## 7. Action buttons

Controls:

- `Run`
- `Open output folder`
- `Open log file`

### 7.1 Run

Behavior before starting:

- Validates all inputs.
- Builds a batch plan.
- Shows the batch plan as running.
- Initializes the progress bar with skipped files already counted.

If validation fails:

- show warning dialog `Invalid input`
- do not start processing

If the batch plan contains zero files to process:

- show information dialog `Nothing to run`
- message states that all selected files are already processed
- status label becomes `All <N> file(s) already processed.`
- no worker thread starts
- controls remain enabled

If processing starts:

- settings are saved
- all input/option controls are disabled for the duration of the run
- `Run` button becomes disabled and its label changes to `Running…`
- processing occurs on a background thread

### 7.2 Open output folder

Default:

- disabled on startup

Becomes enabled:

- after a worker-backed run returns normally, including runs with per-file failures
- not enabled by the `Nothing to run` early-return case

Behavior:

- opens the last run's output directory in the desktop file browser
- only works if the remembered output directory still exists

### 7.3 Open log file

Behavior:

- always available
- opens the current session's log file in the desktop default application

## 8. Status label and live log pane

### 8.1 Status label

Default text:

- `Ready.`

During execution:

- updated with progress messages from the backend, for example:
  - `Initializing pipeline...`
  - `Pre-scan: X to process, Y already done`
  - `[i/n] Processing <file> (overall a/b)...`
  - `Loading echogram from <file> (frames s-e)...`
  - `Echogram loaded: shape=(...), t.s`
  - `Loading model (<checkpoint>)...`
  - `Running inference...`
  - `Done in t.s - saved ...`
  - `Batch finished: X processed, Y skipped, Z failed`

On success:

- `Done: <processed> processed, <skipped> skipped, <failed> failed (<seconds>s)`

On failure:

- `Failed.`

### 8.2 Log pane

Control:

- read-only multi-line text box with placeholder `Log output…`

Behavior:

- shows INFO-and-above log messages from the running app
- updates by polling a log queue every 200 ms
- includes backend progress messages
- also includes `print()` output emitted by pipeline code because stdout is redirected into the logger during execution

## 9. Batch preview and progress panel

This panel is hidden when there is not enough valid information to plan a run.

The panel contains:

- summary label
- segmented progress bar
- legend
- ETA label

### 9.1 When the panel appears

The panel appears only when:

- at least one input file is selected, and
- a valid shared config can be built

The panel is cleared if planning fails because of invalid config, missing checkpoint, or no selected files.

### 9.2 Summary text

Before a run:

- if some files will run and some will skip:
  - `<process_count> file(s) to process, <skip_count> already done (will skip).`
- if all files will skip:
  - `All <skip_count> file(s) already processed — nothing to run.`
- if no files will skip:
  - `<process_count> file(s) to process.`

When a run has started and some files will skip:

- append: `Skipped files are shown in gray on the bar.`

### 9.3 Progress bar

The progress bar is segmented by file count, not by frame count or elapsed time.

Meaning of segments:

- gray: skipped because already processed
- red: failed during this run
- purple: completed during this run
- light gray: pending

If total files is zero:

- display `No files`

Center label format:

- `<skipped> skipped · <completed> done`
- optionally add `· <failed> failed`
- optionally add `· <pending> pending`
- append `(<finished>/<total>)`

### 9.4 ETA

Behavior:

- hidden before a run starts
- visible during runs that have files to process
- shown as `Time remaining: —` until at least one file completes successfully
- then estimated from:
  - elapsed run time
  - number of successfully completed files
  - pending file count
- failed files reduce the pending count
- when no files remain pending, ETA shows `0:00:00`

## 10. Validation rules

Run validation enforces:

- at least one input file selected
- checkpoint file exists
- if multiple files are selected, output directory must be set
- every selected input path must:
  - have suffix `.aris` or `.ddf`
  - exist as a file

Validation messages currently used:

- `Select at least one ARIS or DIDSON file`
- `Checkpoint not found: <path>`
- `Set an output directory when processing multiple files`
- `Expected .aris or .ddf file: <path>`
- `ARIS file not found: <path>`

## 11. Controls disabled while running

When a run is active, the GUI disables:

- ARIS summary field
- `Add files…`
- `Add directory…`
- `Clear`
- entire directory options group
- both directory mode radio buttons
- directory limit spin box
- `Skip already processed`
- output directory field
- output directory `Browse…` button
- start frame
- end frame
- all export checkboxes
- both upstream radio buttons
- advanced toggle
- advanced `Reset all` button
- `Filter submasks`
- `Native FPS`
- manual `Inference FPS`
- `Native inference bins`
- manual `Inference bins`
- `Confidence`
- `IoU`
- `Device`
- checkpoint selector controls

Still active while running:

- `Open log file`
- live status updates
- live log pane

No cancel/stop button exists in the current UI.

## 12. Completion and error dialogs

### 12.1 Successful completion

Dialog title:

- `Complete` for a single selected file
- `Batch complete` for multiple selected files

Dialog body contains:

- processed count
- skipped count
- failed count
- output directory
- optional `Last output:` section pointing to the first exported file path from the last successful file
- optional `Failures:` section listing up to 10 failed files with messages
- if more than 10 failures exist, add `... and <N> more`

### 12.2 Run failure

If the worker thread raises an uncaught exception:

- show critical dialog `Pipeline failed`
- body is the exception text
- re-enable controls
- refresh the batch preview

### 12.3 Startup failure

If the application crashes before the window is usable:

- write a crash log
- show a critical dialog `EchoSeg failed to start`
- include the crash log path

## 13. Saved settings

The GUI persists these values across launches:

- checkpoint path
- whether `Use other checkpoint` is enabled
- selected ARIS file paths
- selected source directory
- directory mode: all vs first N
- directory limit N
- `Skip already processed`
- output directory
- start frame
- end frame
- `Analyse all frames`
- export PNG
- export CSV
- export FC
- export Echotastic
- `Include input echogram`
- `Include summary bar`
- upstream direction
- filter submasks
- confidence
- IoU
- native inference FPS mode
- inference FPS value
- native inference bins mode
- inference bins value
- device
- whether Advanced is expanded

Restore behavior:

- if a source directory was saved, it is rescanned on startup using the saved directory mode
- if a saved checkpoint path is no longer present in the bundled list, the raw path may still be kept in the custom path field

## 14. Output contract

### 14.1 Output directory rules

Resolved output directory for a given file:

- if `Output directory` is set: use that path
- otherwise: use the input file's parent directory

The output directory is created automatically if needed.

### 14.2 Output filenames

For an input file with stem `<stem>`, start frame `<s>`, and end frame `<e>`:

- PNG: `<parent-folder>__<stem>_<s>_<e>_predictions.png`
- CSV: `<stem>_<s>_<e>_predictions.csv`
- FC: `FCe_<stem>_ID_.txt`
- Echotastic: `<stem>.aris.txt`

Notes:

- FC and Echotastic filenames do not include the frame range.
- PNG and CSV filenames do include the frame range.

### 14.3 PNG export

The PNG is an echogram prediction visualization generated by the backend plotting code.

Functional requirement for replacement GUIs:

- the GUI must allow enabling/disabling PNG export
- the GUI must surface the saved path after completion

### 14.4 CSV export

The CSV contains one row per predicted instance with these exact columns:

- `instance_index`
- `confidence`
- `class_id`
- `class_name`
- `enter_frame`
- `exit_frame`
- `center_frame`
- `center_frame_bin`
- `center_frame_distance`
- `duration`
- `minimum_bin_y`
- `maximum_bin_y`
- `average_bin_y`
- `start_bin_y`
- `end_bin_y`
- `minimum_distance`
- `maximum_distance`
- `average_distance`
- `start_distance`
- `end_distance`

### 14.5 FC export

Purpose:

- FishClass manual marking export

Filename:

- `FCe_<arisname>_ID_.txt`

Behavior:

- no-cross detections are excluded
- direction depends on `Upstream`
- output is fixed-width text

### 14.6 Echotastic export

Purpose:

- tab-delimited Echotastic export

Filename:

- `<arisname>.aris.txt`

Behavior:

- no-cross detections are excluded
- direction depends on `Upstream`
- header includes file/date/time metadata when derivable

## 15. Hidden backend settings that are fixed by the UI

The GUI does not expose these backend settings; they are fixed to these values:

- `bgs = True`
- `raw_third_channel = True`
- `fp16 = False`
- `crop_around_gt = False`
- `beam_width_dir = beam_widths_dir()`
- `horizontal_stretch = 1`
- `spacer_width = 0`
- `spacer_colour = (255, 255, 255)`
- `show = False`

If a replacement GUI must match current functionality exactly, these should remain fixed unless intentionally expanded into new controls.

## 16. Platform/logging behavior relevant to UX

Log file behavior:

- each app session creates a timestamped log file
- filename pattern: `echo_seg_YYYYMMDD_HHMMSS.log`

Log storage location:

- development mode: `<repo>/logs/`
- frozen Linux build: `~/.local/share/echo-seg/`
- frozen Windows build: `%LOCALAPPDATA%/echo-seg/`

Crash log path:

- `<log dir>/crash.log`

## 17. Minimum parity requirements for a replacement GUI

A replacement GUI should be considered functionally equivalent only if it preserves:

- all controls and defaults listed above
- directory-vs-file input modes
- skip-already-processed pre-scan behavior
- persisted settings behavior
- disabled-during-run behavior
- background-thread execution
- live status and log streaming
- segmented batch progress with skipped/failed/completed counts
- exact output filename rules
- validation messages of equivalent meaning
- completion/error dialogs with counts and output location
