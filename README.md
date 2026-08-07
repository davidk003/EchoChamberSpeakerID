# EchoChamber — Speaker ID audio ingestion

Low-latency Windows audio capture that turns a live microphone (or a WAV file) into
**configurable overlapping windows**, ready to hand to an audio classification model.
The ML stage is deliberately out of scope: the pipeline ends at a pluggable sink.

## Quick start

```bash
pip install -r requirements-dev.txt
python -m echochamber.app          # GUI: pick a device, start, watch the meters
pytest                             # 600+ tests, no microphone required
```

## What it does

```
 device / WAV ──► RingBuffer ──► WindowChunker ──► bounded queue ──► consumer thread ──► sink
                                                                                          │
                                            GUI polls a stats snapshot at 30 Hz ──────────┘
```

- **Sample-exact windowing.** Chunk `k` covers frames `[k*H, k*H + W)`, so consecutive
  windows share exactly `W - H` identical samples. Read positions come from a counter, never
  a wall clock, so the grid never drifts no matter how the scheduler behaves.
- **Configurable geometry, live.** `window_ms` and `hop_ms` change while capture runs;
  overlap is derived rather than configured, keeping the cadence integral.
- **The audio callback does almost nothing.** Downmix, one memcpy into a preallocated ring,
  bump a counter. No allocation, no locks, no Qt.
- **A slow consumer cannot stall capture.** The bounded queue plus a dedicated consumer
  thread absorb it; drops are counted and surfaced rather than hidden.
- **Testable without hardware.** `FileSource` replays a WAV through the identical path, and
  a fake `sounddevice` covers device enumeration — so the suite runs anywhere.

## Layout

| Path | What lives there |
|---|---|
| `echochamber/config.py` | `AudioConfig`: window/hop geometry, validation |
| `echochamber/audio/ringbuffer.py` | SPSC ring with a doubled-write layout (contiguous reads) |
| `echochamber/audio/chunker.py` | The windowing thread |
| `echochamber/audio/sinks.py` | `QueueSink`, `TeeSink`, `WavRecorderSink`, latency tracking |
| `echochamber/audio/sources/` | `FileSource` (replay) and `SoundDeviceSource` (live) |
| `echochamber/audio/pipeline.py` | Wires it together |
| `echochamber/voicegate/` | Wake-phrase gate: record a snippet only when a phrase is said |
| `echochamber/ui/` | PySide6 GUI; all logic in `controller.py`, widgets stay dumb |
| `docs/architecture.md` | Design rationale, latency model, and the hard-won gotchas |

## Wake-phrase voice gate

Off by default. When enabled, the pipeline stops keeping everything and keeps only what
follows a configured phrase — `"ok google"`, `"hey google"` — as one WAV snippet each.

```bash
python scripts/setup_voice_gate.py     # fetches the model, builds the recognizer venv
```

It is a `ChunkSink` composed alongside the existing one, so nothing upstream changes. The
snippet is seeded with `pre_roll_ms` of audio kept from *before* the match, because a
recognizer only reports a phrase after consuming it — without the pre-roll the snippet
would start after the phrase it is named for. Matching is on whole words, so `"ok google"`
fires on `"ok google turn it up"` but not on `"look google it"`.

**`vosk` has no `win_arm64` wheel**, and the deployment target is Windows ARM64. So on
ARM64 vosk is not installed into the main environment at all: it runs in a separate
**x64** venv as a subprocess (Windows' Prism emulator runs it transparently) and the gate
talks to it over a pipe, keeping the real-time capture path native. `setup_voice_gate.py`
prints exactly what to configure. See `docs/architecture.md` §3.7 for the full rationale —
including what has *not* yet been verified on real ARM64 hardware.

Recognition is pluggable and absent by default: with no vosk and no model, the gate is
inert and the entire test suite still passes.

### Announcing detections

The gate can push events to a WebSocket listener (`pip install .[notify]`):

```python
NotifyConfig(enabled=True, url="ws://127.0.0.1:8765", include_audio=True)
```

Two events per utterance. `detected` goes out **the moment the phrase is recognized**;
`snippet` follows when the file closes — `post_roll_ms` later — and carries the WAV when
`include_audio` is set. They share a `seq`, so a listener can pair them:

```json
{"type":"detected","phrase":"ok google","text":"ok google turn it up","seq":0,"sample_rate":16000,"timestamp":1786122719.665}
{"type":"snippet","phrase":"ok google","seq":0,"path":"...","frames":16000,"duration_s":1.0,"truncated":false,"audio":{"encoding":"base64","format":"wav","bytes":32044,"data":"..."}}
```

Sending happens on its own thread behind a bounded queue — a socket write to a dead host
takes a TCP timeout to fail, and doing that on the consumer thread would stall the
pipeline and drop audio. A listener that cannot keep up costs counted, visible dropped
events instead. Best-effort only: no acknowledgement, no redelivery, and queued events are
discarded at shutdown.

## Two things that will bite you

**WASAPI will not resample unless you ask it to.** A 16 kHz request on a 48 kHz device fails
with `Invalid sample rate [PaErrorCode -9997]` unless the stream is opened with
`WasapiSettings(auto_convert=True)`. Nearly every Windows capture device is 44.1 or 48 kHz.

**After an overrun, `start_frame` is not a multiple of `hop`.** The grid restarts wherever
surviving audio begins. Use `seq`, which is gap-free; such chunks are flagged `discontinuous`.

Both are covered in more detail in `docs/architecture.md`.

## Platform

Developed on Windows x86-64, **deployed to Windows ARM64**. Every dependency with a compiled
extension needs a verified `win_arm64` wheel before adoption.
