# Audio Ingestion Architecture

Scope: capture live audio, cut it into **configurable overlapping windows**, and hand those
windows to a downstream consumer with minimal added latency. ML inference is out of scope —
the pipeline terminates at a pluggable sink.

Target platform: Windows, Python 3.11, PySide6 GUI.

> **Deployment constraint.** Development happens on x86-64, but the deployment target is
> **Windows ARM64**. The dev machine will not surface ARM64 packaging failures, so every
> dependency with a compiled extension must be checked for a `win_arm64` wheel *before*
> adoption. This already ruled out `soxr` (no ARM64 wheel) and is why `sounddevice` was
> chosen (0.5.5 ships `win_arm64`).
>
> Unresolved, out of scope until the ML stage exists: the dev environment has
> `torch 2.10.0+xpu`, an **Intel XPU build that is x86-only**. Inference on ARM64 will need a
> different runtime — ARM64 CPU torch, or ONNX Runtime, which has solid Windows ARM64 support.

---

## 1. Decisions

| Decision | Choice | Why |
|---|---|---|
| Capture backend | `sounddevice` (PortAudio, WASAPI) | Callback-driven, low latency, mature. Ships a `win_arm64` wheel as of 0.5.5. |
| Source | Microphone / input devices | Loopback deferred; the source layer is an interface so it can be added later. |
| Sample rate | Request target rate (16 kHz) from the device | WASAPI shared mode resamples inside the audio engine. Avoids a resampler dependency — `soxr` has **no ARM64 wheel**. Optional high-quality path behind a flag. |
| Format | mono `float32` | What every audio model wants. Downmix happens once, in the callback. |
| Handoff | In-process bounded queue + worker thread | Lowest latency. torch releases the GIL during inference, so a thread is sufficient. |
| Sink API | `ChunkSink` protocol | Lets the queue be swapped for a process/network sink later without touching capture. |

**Non-negotiable rule:** the PortAudio callback runs on a real-time-ish thread. It must never
allocate, never lock, never log, and never touch Qt. Violating this causes xruns (dropouts),
not just slowness.

---

## 2. Data flow

```
 PortAudio callback thread  (RT, ~10 ms blocks)
        │  downmix to mono, memcpy into ring, bump write cursor, set event
        ▼
   RingBuffer  (preallocated float32, SPSC, lock-free)
        │  blocking read of W samples once available
        ▼
 Chunker thread            (windowing: window W, hop H)
        │  emits AudioChunk(samples, start_frame, seq, sample_rate)
        ▼
   TeeSink ──┬──► QueueSink  (bounded, maxsize ~8) ──► consumer thread → [future ML]
             ├──► WavRecorderSink  (optional, debug / ground truth)
             └──► MeterSink  (updates an atomic stats snapshot)
                          ▲
 GUI thread ──────────────┘  QTimer @ 30 Hz polls the snapshot. No per-chunk signals.
```

Three threads, one queue, one ring buffer. Nothing else crosses a thread boundary.

---

## 3. Component detail

### 3.1 RingBuffer (`audio/ringbuffer.py`)

Single-producer / single-consumer, preallocated `np.empty(capacity, float32)`. Capacity ≈ 10 s,
which is ~640 KB at 16 kHz — cheap, and large enough to absorb a GUI hitch without loss.

**Contiguous-read trick:** allocate `2 * capacity` and write every block twice (at `i` and
`i + capacity`). Reads of any window ≤ capacity are then always a contiguous slice — no
wraparound branch, no stitching, and the chunker can hand out a plain `np.ndarray` view.
Cost is one extra memcpy of a 10 ms block in the callback; negligible.

State shared across threads is exactly one monotonically increasing `write_frames` counter
(a Python int assignment, which is atomic under the GIL). The consumer never writes it.

Overrun detection: if the reader falls more than `capacity` behind the writer, samples were
lost. Detect on read, count it, resynchronize the read cursor to the oldest valid sample, and
mark the next chunk `discontinuous=True` so downstream knows not to trust continuity.

### 3.2 Chunker (`audio/chunker.py`)

The core of the feature. Two config values, both in milliseconds:

- `window_ms` → `W` samples: how much audio each chunk contains.
- `hop_ms` → `H` samples: how far the window advances. **Overlap is derived**: `W - H`.
  Expressing it as hop rather than overlap-percent keeps chunk cadence exact and integer.

Chunk `k` covers absolute frames `[k*H, k*H + W)`. The loop is:

```
next_start = 0
while running:
    wait until ring.write_frames >= next_start + W
    samples = ring.read(next_start, W)        # contiguous view, then copy
    emit AudioChunk(samples, start_frame=next_start, seq=k, ...)
    next_start += H
```

Because `start_frame` is derived from a counter and never from a wall clock, chunk boundaries
are **sample-exact and drift-free** regardless of scheduler jitter.

**Copy semantics:** `read()` returns a view into the ring; the chunker copies it before
emitting, because the writer will eventually overwrite that region. For W = 3 s @ 16 kHz that
is a 192 KB memcpy per hop — microseconds. Do not try to avoid it.

**Live reconfiguration:** `window_ms` / `hop_ms` live in a frozen dataclass held in a single
attribute; the GUI swaps the whole object atomically. The chunker re-reads it at the top of
each iteration, realigns `next_start` to the current write head, and flags the next chunk
`discontinuous=True`. No locks, no restart of the audio device.

### 3.3 Latency model

This is what determines whether the pipeline feels real-time. Two distinct numbers:

- **Chunk cadence** = `hop_ms`. How often a result arrives.
- **Result latency** = `window_ms` + overhead. A chunk cannot be emitted until its *last*
  sample has been captured, so the newest audio inside chunk `k` is `hop` old but the oldest
  is `window` old.

Overhead budget beyond `window_ms`:

| Stage | Cost |
|---|---|
| WASAPI shared-mode device buffer | ~10 ms (blocksize 160 @ 16 kHz) |
| Ring write + read | < 0.1 ms |
| Chunker wake-up jitter | ~1–2 ms (Windows timer granularity) |
| Queue put/get | < 0.1 ms |
| **Total added** | **~12 ms** |

So a 3000 ms window / 1000 ms hop config produces a chunk every second, each ~3.01 s behind
the leading edge of the *oldest* audio it contains. If perceived latency is too high, the lever
is `window_ms` (bounded by how much context the model needs), not the transport.

Instrument this: stamp `capture_time` on the chunk at emit and let the consumer record
`now - capture_time` into a histogram surfaced in the GUI. Measure, don't assume.

### 3.4 Backpressure (`audio/sinks.py`)

`QueueSink` wraps a `queue.Queue(maxsize=8)` with an explicit policy:

- `DROP_OLDEST` (default, live): if full, discard the head and enqueue the new chunk. Freshness
  beats completeness for real-time classification. Increment `dropped_chunks`.
- `BLOCK` (offline/file replay): back-pressures the chunker, which back-pressures nothing
  (the ring keeps filling and eventually overruns) — so this is only correct for file sources.

The drop counter must be visible in the GUI. A silently-dropping pipeline that looks healthy is
the single most common failure mode in this kind of system.

### 3.5 Sources (`audio/sources/`)

`AudioSource` interface: `start(callback)`, `stop()`, `sample_rate`, `channels`.

- `SoundDeviceSource` — live capture. Also owns device enumeration and xrun counting via
  PortAudio's `status` flags (`input_overflow`).
- `FileSource` — replays a WAV through the identical ring/chunker path, either at real-time
  pace or as fast as possible. This is what makes the chunker **deterministically testable**
  and lets the whole GUI be exercised without a microphone.
- `QtAudioSource` — thin `QAudioSource` fallback, only if a platform turns up where PortAudio
  misbehaves. Not built initially.

### 3.6 GUI (`ui/`)

PySide6. The audio path never calls into Qt directly.

- A 30 Hz `QTimer` polls a `StreamStats` snapshot (peak/RMS level, chunk count, dropped count,
  xruns, measured latency). Polling rather than per-chunk signals — at `hop_ms = 50` a signal
  per chunk would flood the event loop, and the display can't show more than ~60 Hz anyway.
- Rare events (device error, stream stopped, discontinuity) do use queued `Signal`s.
- Widgets: device selector, start/stop, `window_ms` / `hop_ms` spinboxes with a live "overlap:
  N ms (P %)" readout, level meter, scrolling waveform, and a stats panel showing dropped
  chunks and the latency histogram.

Waveform rendering pulls a decimated view from the ring buffer on the timer tick — the GUI is
a *reader* of the same ring, never a sink that can block the pipeline.

---

## 4. Module layout

```
echochamber/
  config.py              # AudioConfig frozen dataclass, defaults, ms→samples helpers
  audio/
    types.py             # AudioChunk, StreamStats, DropPolicy
    ringbuffer.py        # SPSC ring, doubled-write contiguous reads
    chunker.py           # WindowChunker thread
    sinks.py             # ChunkSink protocol, QueueSink, TeeSink, WavRecorderSink
    devices.py           # input device enumeration
    pipeline.py          # AudioPipeline: wires source→ring→chunker→sinks, start/stop
    sources/
      base.py
      sounddevice_source.py
      file_source.py
  ui/
    main_window.py
    device_panel.py
    meters.py
    stats_panel.py
  app.py                 # entry point
tests/
  test_ringbuffer.py
  test_chunker.py
  test_pipeline.py
```

## 5. Defaults

```python
sample_rate = 16_000     # what speech models expect
channels    = 1          # downmix in the callback
dtype       = float32
blocksize   = 160        # 10 ms device callback
ring_seconds = 10.0
window_ms   = 3000
hop_ms      = 1000       # → 2000 ms (67 %) overlap
queue_max   = 8
drop_policy = DROP_OLDEST
```

## 6. Testing

The chunker is pure logic over a buffer, so it gets tested exactly:

- Feed a **sample-index ramp** (`samples[i] == i`) through `FileSource`. Assert chunk `k`
  starts with value `k*H`, has length `W`, and that consecutive chunks overlap by exactly
  `W - H` identical samples. This catches off-by-one errors that are invisible on real audio.
- Ring buffer: wraparound correctness, overrun detection, contiguity of reads spanning the seam.
- Reconfiguration mid-stream: assert alignment resets and `discontinuous` is set once.
- Backpressure: a deliberately slow sink must drop, not block, and the drop count must match.

## 7. Dependencies to add

- `sounddevice >= 0.5.5` (win_arm64 wheel available)
- `pytest`

Deliberately *not* added: `soxr` / `librosa` / `scipy` for resampling — no ARM64 wheels and not
needed while the device delivers the target rate. Revisit only if measured resampling quality
becomes a problem.

## 8. Build order

1. ~~`types.py`, `config.py`, `ringbuffer.py` + tests~~ — **done** (288 tests passing)
2. `chunker.py` + ramp tests (no audio device needed yet)
3. `sinks.py`, `pipeline.py`, `file_source.py` — full pipeline testable headless
4. `sounddevice_source.py` + device enumeration — first live audio
5. GUI: device panel, start/stop, meters, stats
6. Latency instrumentation + a stub consumer standing in for the ML stage
```
