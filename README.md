# Echo Seg Web

Browser-first local app for decoding ARIS/DIDSON files and running ONNX segmentation entirely in the browser.

## Development

```bash
npm install
npm run dev
```

The Vite app lives under [`web/`](./web).

## Production build

```bash
npm run build
```

The static bundle is written to `dist-web/`.

## Features

- local single-file `.aris` / `.ddf` decode in JavaScript
- local ONNX inference with `onnxruntime-web`
- decoded echogram preview and detection overlay
- CSV / FC / Echotastic exports from the browser UI
- bundled-model support via `weights/noklamath.onnx`

## Decoder parity check

The browser decoder can be checked against the legacy Python decoder implementation:

```bash
pip install -r requirements.txt
npm run verify:decoder
```

By default that runs against `example.aris` and compares the decoded bytes exactly.

## Layout

| Path | Role |
| --- | --- |
| `web/` | Vite app source |
| `fisheye_loading/` | Python decoder used for parity verification |
| `beam_widths/` | ARIS beam-width CSV inputs for the Python decoder |
| `scripts/` | Webapp-adjacent utility scripts |
| `weights/` | Browser ONNX models |
