# EchoSeg UI Quick Reference

## What This App Does

The desktop app runs the EchoSeg model on one or more ARIS / DIDSON sonar files (`.aris` or `.ddf`).

In plain terms, it:

- loads sonar files
- runs the model
- can skip files whose selected outputs already exist
- writes one or more output file types
- shows a batch preview before running
- shows progress, ETA, status, and logs during a run

Possible outputs:

- PNG overlay image
- CSV predictions table
- FishClass FC text file
- Echotastic `.aris.txt` export

## Required Before You Click Run

| Input | Required? | Notes |
| --- | --- | --- |
| ARIS input | Yes | Select at least one `.aris` or `.ddf` file, or a directory containing them. |
| Checkpoint | Yes | Must point to an existing `.pt` model file. You can use a bundled checkpoint or a custom one. |
| Output directory | Only for batch runs | Required when processing multiple files. Optional for a single file. |

## Main Inputs

### Inputs

| Control | Required? | Default | What it does |
| --- | --- | --- | --- |
| `ARIS input` | Yes | Empty | Shows the current file or directory-based selection. |
| `Add files…` | One input method required | N/A | Lets you pick one or more `.aris` / `.ddf` files directly. Files are appended, de-duplicated, and sorted A-Z. |
| `Add directory…` | One input method required | N/A | Lets you pick a directory and build the file list from its top-level `.aris` / `.ddf` files. |
| `Clear` | Optional | N/A | Clears the current input selection and exits directory mode. |
| `All files in directory (A-Z)` | Optional | On | In directory mode, process every `.aris` / `.ddf` file in alphabetical order. |
| `First N files in directory (A-Z)` | Optional | Off | In directory mode, process only the first `N` files alphabetically. |
| `N` | Optional | `10` | Active only when `First N files...` is selected. Range: `1` to `1,000,000`. |
| `Skip already processed` | Optional | On | Skips files whose currently enabled outputs already exist. |
| `Output directory` | Conditionally required | Auto-filled when possible | Destination for outputs. If blank on a single-file run, outputs go beside the input file. |

### Model

| Control | Required? | Default | What it does |
| --- | --- | --- | --- |
| `Checkpoints` | Yes, unless using custom checkpoint | First bundled checkpoint if available | Selects a bundled `.pt` model file. |
| `Use other checkpoint` | Optional | Off | Switches from bundled checkpoint selection to a custom file path. |
| `Other checkpoint` | Required only when `Use other checkpoint` is on | Empty or copied from current bundled selection | Path to a custom `.pt` model file. |

### Options

| Control | Required? | Default | What it does |
| --- | --- | --- | --- |
| `Upstream: Left / Right` | Optional | `Left` | Sets the direction mapping used by FC and Echotastic exports. |
| `Analyse all frames` | Optional | On | If on, the full file is processed and the manual frame range is disabled. |
| `Start` | Optional | `0` | First frame to process. Only used when `Analyse all frames` is off. |
| `End` | Optional | `-1` | Exclusive end frame. `-1` means end of file. Only used when `Analyse all frames` is off. |
| `Export PNG` | Optional | Off | Saves a PNG visualization of the predictions. |
| `Include input echogram` | Optional | On | Only applies when `Export PNG` is enabled. Includes the input echogram in the PNG output. |
| `Include summary bar` | Optional | On | Only applies when `Export PNG` is enabled. Adds the summary bar to the PNG output. |
| `Export CSV` | Optional | On | Saves prediction results as a CSV file. |
| `Export FC files` | Optional | On | Saves FishClass-compatible `FCe_<arisname>_ID_.txt` output. |
| `Export Echotastic` | Optional | Off | Saves tab-delimited `<arisname>.aris.txt` output. |

### Advanced

| Control | Required? | Default | What it does |
| --- | --- | --- | --- |
| `Advanced` | Optional | Collapsed | Expands or collapses the advanced settings section. |
| `Reset all` | Optional | N/A | Resets every advanced control back to its default value. |
| `Filter submasks` | Optional | On | Filters model submasks before export. |
| `Confidence` | Optional | `0.10` | Minimum detection confidence threshold. Range: `0.00` to `1.00`. |
| `IoU` | Optional | `0.50` | Non-max suppression IoU threshold. Range: `0.00` to `1.00`. |
| `Native FPS` | Optional | On | Uses native FPS behavior for inference width rescaling. |
| `Inference FPS` | Optional | Disabled until `Native FPS` is unticked | Manual FPS value used only when `Native FPS` is off. Range: `1.0` to `120.0`. |
| `Native inference bins` | Optional | On | Uses native inference bins instead of a fixed width preset. |
| `Inference bins` | Optional | Disabled until `Native inference bins` is unticked; manual default `1280` | Manual width preset used only when `Native inference bins` is off. Options: `640`, `1280`, `1920`, `2560`. |
| `Device` | Optional | `cpu` | Compute device. CUDA devices appear if available. |

## Running And Monitoring

| Control / Area | Default | What it does |
| --- | --- | --- |
| `Run` | Enabled | Validates inputs, builds the batch plan, and starts the pipeline. While running it becomes disabled and its label changes to `Running…`. |
| `Open output folder` | Disabled until a run finishes normally | Opens the last run's output directory if it still exists. |
| `Open log file` | Enabled | Opens the current session log file. |
| Status label | `Ready.` | Shows current progress text and final success/failure status. |
| Log pane | Empty with placeholder `Log output…` | Shows live INFO-and-above log messages and backend progress updates. |

## Batch Preview

When enough valid information is present, the app shows:

- a summary of how many files will process vs skip
- a segmented progress bar
- a legend
- an ETA label during runs

Progress colors:

- gray: already processed / skipped
- purple: completed this run
- red: failed
- light gray: pending

## Output Files

Depending on which export toggles are enabled, the app can write:

- PNG: `<parent-folder>__<stem>_<start>_<end>_predictions.png`
- CSV: `<stem>_<start>_<end>_predictions.csv`
- FC: `FCe_<stem>_ID_.txt`
- Echotastic: `<stem>.aris.txt`

## Important Notes

- For batch runs, set an output directory explicitly.
- `End` is exclusive, not inclusive.
- The batch preview can appear before full run validation passes; for example, a multi-file selection can still need an output directory before `Run` succeeds.
- The app currently allows all export toggles to be turned off, but that usually is not useful because a run may finish without writing result files.
- `Reset all` only resets the Advanced section, not the main input/output options.
- Most settings are saved and restored the next time the app opens.
