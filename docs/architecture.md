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
| Sample rate | Request target rate (16 kHz) from the device, **with WASAPI auto-convert enabled** | The Windows audio engine resamples for us. Avoids a resampler dependency — `soxr` has **no ARM64 wheel**. See the warning below: this does *not* work by default. |
| Format | mono `float32` | What every audio model wants. Downmix happens once, in the callback. |
| Handoff | In-process bounded queue + worker thread | Lowest latency. torch releases the GIL during inference, so a thread is sufficient. |
| Sink API | `ChunkSink` protocol | Lets the queue be swapped for a process/network sink later without touching capture. |

**Non-negotiable rule:** the PortAudio callback runs on a real-time-ish thread. It must never
block, never log, and never touch Qt. Keep allocation to the unavoidable minimum: in practice
the callback does one `copy()` of the device block, computes peak/RMS, and calls
`Event.set()` (which briefly takes an uncontended lock) — measured at 0 xruns over multi-second
runs, but it is not literally allocation-free, and anything heavier belongs downstream.
Violating the spirit of this causes xruns (dropouts), not just slowness.

> **WASAPI will not resample unless you ask it to.** Measured on real hardware (Intel mic
> array, 48 kHz native, PortAudio 19.7):
>
> ```
> 16000 Hz mono, plain        -> PortAudioError: Invalid sample rate [PaErrorCode -9997]
> 16000 Hz mono, auto_convert -> opens at 16000 Hz, real signal
> ```
>
> Requesting a non-native rate from a WASAPI device fails outright unless the stream is opened
> with `sounddevice.WasapiSettings(auto_convert=True)` passed as `extra_settings`. Nearly every
> Windows capture device is 44.1 or 48 kHz, so without this flag 16 kHz capture fails on
> essentially all of them. MME and DirectSound convert without the flag; WASAPI is the one that
> needs it, and WASAPI is what we want for latency. Never silently fall back to the device's
> native rate — the rest of the pipeline derives window geometry from `config.sample_rate`, so a
> mismatch corrupts every timestamp rather than merely sounding wrong.

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
   QueueSink  (bounded, maxsize ~8)
        │  consumer thread drains it
        ▼
   sink   (one ChunkSink; use TeeSink to fan out to a recorder, a stub consumer, …)
                          ▲
 GUI thread ──────────────┘  QTimer @ 30 Hz polls a stats snapshot. No per-chunk signals.

Levels are computed in `AudioSource._emit`, the only place raw audio is seen; there is no
separate meter sink. `AudioPipeline` wires exactly one sink — `TeeSink` is available but not
used by default.
```

Three threads, one queue, one ring buffer. Nothing else crosses a thread boundary.

---

## 3. Component detail

### 3.1 RingBuffer (`audio/ringbuffer.py`)

Single-producer / single-consumer, preallocated `np.zeros(2 * capacity, float32)` — twice the
nominal capacity because of the doubled-write layout below. A 10 s ring at 16 kHz is therefore
~1.28 MB, still cheap, and large enough to absorb a GUI hitch without loss.

**Contiguous-read trick:** allocate `2 * capacity` and write every block twice (at `i` and
`i + capacity`). Reads of any window ≤ capacity are then always a contiguous slice — no
wraparound branch, no stitching, and the chunker can hand out a plain `np.ndarray` view.
Cost is one extra memcpy of a 10 ms block in the callback; negligible.

The hot path shares exactly one monotonically increasing `write_frames` counter (a Python int
assignment, atomic under the GIL); the consumer never writes it. A closed flag and an `Event`
are also shared, but only on the shutdown path, and both must be re-checked after
`Event.clear()` — missing that on the flag once left `wait_for(timeout=None)` able to block
forever on an already-closed ring.

Overrun detection: if the reader falls more than `capacity` behind the writer, samples were
lost. Detect on read, count it, resynchronize the read cursor to the oldest valid sample, and
mark the next chunk `discontinuous=True` so downstream knows not to trust continuity.

**Validating before the copy is not enough.** `read()` returns a view, so its check is already
stale when it returns: a writer that laps the ring while the consumer is copying yields the
wrong audio under a `start_frame` that does not describe it, with nothing raised. Silently
misaligned windows are worse than missing ones — nothing downstream can detect them. `read_copy()`
is therefore a seqlock read: copy, then re-check that the region survived, and raise `OverrunError`
if it did not. Anything that keeps its data must use `read_copy`, never `read().copy()`.

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
    samples = ring.read_copy(next_start, W)   # seqlock read: copy, then re-verify
    emit AudioChunk(samples, start_frame=next_start, seq=k, ...)
    next_start += H
```

Because `start_frame` is derived from a counter and never from a wall clock, chunk boundaries
are **sample-exact and drift-free** regardless of scheduler jitter.

**Copy semantics:** the chunker uses `read_copy()`, which copies and then re-verifies that the
writer did not lap the region mid-copy (see §3.1). For W = 3 s @ 16 kHz that is a 192 KB memcpy
per hop — microseconds. Do not try to avoid it, and do not downgrade it to `read().copy()`.

**Live reconfiguration:** `window_ms` / `hop_ms` live in a frozen dataclass held in a single
attribute; the GUI swaps the whole object atomically. The chunker re-reads it at the top of
each iteration, realigns `next_start`, and flags the next chunk `discontinuous=True`. No
locks, no restart of the audio device.

The realignment frame is captured by `reconfigure()` at the moment of the call, *not* sampled
by the loop when it notices. The loop may not run for up to `POLL_INTERVAL_S`, and on a live
stream the write head moves the whole time — sampling it in the loop would put the new window
grid at a position determined purely by thread scheduling. That is both surprising to callers
and impossible to test deterministically. Audio buffered between the old read position and the
capture point is deliberately skipped: reconfiguring means "start fresh from now", not "catch
up first".

**Two invariants downstream must not assume.** After an overrun the chunker resyncs to
`ring.oldest_frame`, so `start_frame` is *not* a multiple of `hop_frames` in general — the
grid restarts wherever surviving audio begins. Any consumer deriving chunk index from
`start_frame / hop` is wrong; use `seq`, which is gap-free and never resets. Every such chunk
carries `discontinuous=True`, which is the intended signal.

**Config/ring pairing.** `AudioConfig.ring_frames` is a derived figure; the actual `RingBuffer`
is constructed separately, and nothing forces them to agree. `WindowChunker` rejects a window
larger than the ring's capacity at construction and on `reconfigure()`, because otherwise the
mismatch surfaces as a `ValueError` deep on the chunker thread and the stream simply never
produces a chunk.

### 3.3 Latency model

This is what determines whether the pipeline feels real-time. Two distinct numbers:

- **Chunk cadence** = `hop_ms`. How often a result arrives.
- **Result latency** = `window_ms` + overhead. A chunk cannot be emitted until its *last*
  sample has been captured, so the newest audio inside chunk `k` is `hop` old but the oldest
  is `window` old.

Overhead budget beyond `window_ms`:

| Stage | Cost |
|---|---|
| WASAPI shared-mode device buffer | ~22 ms measured (blocksize 160 @ 16 kHz) |
| Ring write + read | < 0.1 ms |
| Chunker wake-up jitter | ~1–2 ms (Windows timer granularity) |
| Queue put/get | < 0.1 ms |
| **Total added** | **~25 ms measured** |

So a 3000 ms window / 1000 ms hop config produces a chunk every second, each ~3.01 s behind
the leading edge of the *oldest* audio it contains. If perceived latency is too high, the lever
is `window_ms` (bounded by how much context the model needs), not the transport.

Instrument this: `AudioChunk.capture_time` is stamped at emit and the consumer records
`now - capture_time` into a `LatencyTracker`, whose percentiles the GUI shows. Measure, don't
assume.

**Measured on real hardware** (live WASAPI mic, 16 kHz, 400 ms window / 200 ms hop):

```
pipeline p50   0.1 ms      <- consumer handoff: queue + thread scheduling
pipeline p95   2.1 ms
device latency 22.2 ms     <- WASAPI input buffer
```

So the transport really is a couple of milliseconds and the window length really is the whole
story, as predicted above. Note what `capture_time` measures: the window has *already* been
filled when it is stamped, so these figures are the handoff cost, not `window_ms + handoff`.

> **Use `time.perf_counter()`, never `time.monotonic()`, for these timestamps.** On Windows
> `time.monotonic()` is `GetTickCount64` with **15.625 ms** resolution. Measuring a
> sub-millisecond handoff with it produced alternating 0 ms and 16 ms readings — noise
> perfectly disguised as real latency spikes, and the first version of this instrumentation
> reported exactly that. `perf_counter` is `QueryPerformanceCounter` (~100 ns) and is equally
> monotonic. The two clocks have unrelated origins, so they must never be mixed.

Percentiles, not averages: a mean hides the occasional stall a user actually notices. The
tracker also keeps `worst_ever_ms` outside the rolling window, because a spike that scrolled
out of view still happened.

### 3.4 Backpressure (`audio/sinks.py`)

`QueueSink` is a bounded buffer (a `collections.deque` plus a `Condition` — `queue.Queue` cannot
discard its own head, which `DROP_OLDEST` requires) with an explicit policy:

- `DROP_OLDEST` (default, live): if full, discard the head and enqueue the new chunk. Freshness
  beats completeness for real-time classification. Increment `dropped_chunks`.
- `BLOCK` (offline/file replay): back-pressures the chunker, which back-pressures nothing
  (the ring keeps filling and eventually overruns) — so this is only correct for file sources.

The drop counter must be visible in the GUI. A silently-dropping pipeline that looks healthy is
the single most common failure mode in this kind of system.

### 3.5 Sources (`audio/sources/`)

`AudioSource` interface: `start()`, `stop()`, `sample_rate`, `channels`, `finished`. The audio
callback and the shared `StreamStats` are injected at construction.

The pipeline builds its source through a `SourceFactory`, which it calls with **both**
`ring.write` and its own `StreamStats`. Passing the writer is what keeps a source ignorant of
the ring. Passing the stats is not optional bookkeeping: the source is the only component that
ever sees raw audio, so it is the only one that can fill in `frames_captured`, `peak_level` and
`rms_level`. An earlier design left the caller to thread the same instance into both places,
which failed silently — the pipeline ran perfectly and the meters simply never moved, looking
like a dead input device rather than a wiring mistake.

- `SoundDeviceSource` — live capture. Also owns device enumeration and xrun counting via
  PortAudio's `status` flags (`input_overflow`). Opens WASAPI devices with
  `WasapiSettings(auto_convert=True)`; see the warning in §3.
  **Prefer a WASAPI device in the picker over PortAudio's reported default.** On the dev
  machine the system default input is an *MME* device: it works (MME converts without a flag)
  but measured ~30 ms input latency against ~22 ms for the same physical microphone via
  WASAPI, and it bypasses the auto-convert path entirely, so a device-picker default of "system
  default" would silently never exercise it. MME also truncates device names to 31 characters,
  so the same microphone appears under different names across host APIs — which is why
  `DeviceInfo.label` includes the host API and name lookups reject ambiguous matches.
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
  Note `peak_level` / `rms_level` are *last block only*, not a running maximum, so the meter
  needs its own decay/hold — read naively at shutdown it shows ~0 even after a loud session.
  `PeakHold` does this, and takes `now` as an argument rather than reading the clock, which is
  what makes its decay testable rather than timing-dependent.

  **The poll timer must be a `Qt.PreciseTimer`.** With Qt's default `CoarseTimer` the 33 ms
  interval gets rounded to the Windows scheduler tick and measured **21 Hz, not 30** — a third
  of the frames silently missing, with nothing to indicate why. Measured 30.1 Hz with
  `PreciseTimer`.
- Rare events (device error, stream stopped, discontinuity) do use queued `Signal`s.
- Widgets: device selector, start/stop, `window_ms` / `hop_ms` spinboxes with a live "overlap:
  N ms (P %)" readout, level meter, scrolling waveform, and a stats panel showing dropped
  chunks and the latency histogram.

Waveform data is decimated in `LatestChunkSink.on_chunk` on the consumer thread, from the
chunk's own samples, and the GUI reads the latest snapshot on its timer tick. The GUI never
touches the ring buffer — keeping it a single-consumer structure, which is what the lock-free
design depends on.

---

## 4. Module layout

```
echochamber/
  config.py              # AudioConfig frozen dataclass, defaults, ms→samples helpers
  audio/
    types.py             # AudioChunk (incl. capture_time), StreamStats, DropPolicy
    ringbuffer.py        # SPSC ring, doubled-write contiguous reads, seqlock read_copy
    latency.py           # LatencyTracker / LatencySummary percentiles
    stub_consumer.py     # StubInferenceSink: stands in for the ML stage
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
2. ~~`chunker.py` + ramp tests (no audio device needed yet)~~ — **done** (353 tests passing)
3. ~~`sinks.py`, `pipeline.py`, `file_source.py` — full pipeline testable headless~~ —
   **done** (506 tests passing; a WAV replays end-to-end through the real ring, chunker,
   bounded queue and consumer thread)
4. ~~`sounddevice_source.py` + device enumeration — first live audio~~ — **done**
   (545 tests passing; verified against real hardware at 16 kHz from a 48 kHz WASAPI device)
5. ~~GUI: device panel, start/stop, meters, stats~~ — **done** (656 tests passing; verified
   driving a live microphone at 30.1 Hz with no thread outliving the window)
6. ~~Latency instrumentation + a stub consumer standing in for the ML stage~~ — **done**
   (690 tests passing; `LatencyTracker`, `StubInferenceSink`, percentiles in the stats panel)
```
