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
| `echochamber/ui/` | PySide6 GUI; all logic in `controller.py`, widgets stay dumb |
| `docs/architecture.md` | Design rationale, latency model, and the hard-won gotchas |

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
