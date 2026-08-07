# EchoChamber

**Your phone only listens for "Hey Google" when it hears *your* voice.**

EchoChamber watches the mic for a wake phrase, checks whether it was *you* who said it,
and if it wasn't — blocks Google Assistant from responding at all, live, over ADB. Say it
yourself and everything stays exactly as if EchoChamber weren't there.

```
 mic ──► "hey google"? ──► is that YOUR voice? ──► yes: do nothing
                                    │
                                    └── no: block "Ok Google" on your phone
```

No wake word ever leaves your machine. No cloud, no API keys, nothing installed on the
phone — just `adb` flipping a permission on and off.

## Try it in 2 minutes

```bash
git clone <this-repo-url>
cd EchoChamberSpeakerID
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m echochamber.app
```

Pick a microphone, hit **Start**. You're now looking at a live audio meter — that's the
whole pipeline running with every extra feature off. Everything below is one checkbox
away.

## Turning on the full demo

Three switches, top to bottom, each one unlocking the next:

### 1. Wake-phrase detection (Vosk)

```bash
pip install .[voice-gate]              # skip this line on ARM64 -- see "Platform" below
python scripts/setup_voice_gate.py     # downloads the model, ~50 MB, one time
```

Back in the app, tick **"Enable wake-phrase gate"** in the Voice gate panel. Restart
capture. Say "ok google" or "hey google" out loud — you'll see it counted as a detected
phrase and a short WAV snippet saved to `snippets/`.

### 2. Enroll your voice (speaker ID)

One-time model setup:

```bash
python scripts/setup_speakerid_qnn.py --skip-arm64
.venv-speakerid-x64\Scripts\python.exe scripts\export_speakerid_qnn.py
```

Then enroll — right from the app, no terminal needed. In the new **Speaker ID & ADB
hotword trigger** panel:

- Type your name.
- **Have a WAV file of your voice already?** Click **"Choose WAV file..."**, pick it, hit
  **"Enroll from file"**. This is the easy, reliable way — a clean few seconds of just you
  talking, recorded however you like.
- **No file handy?** Use **"Record and enroll"** just below instead — it grabs a few
  seconds straight from your microphone.

Tick **"Require my voice to open a snippet"**. Restart capture. Say the wake phrase
yourself → snippet. Have someone else say it → nothing, silently suppressed.

### 3. Block the real Assistant hotword (ADB)

This is the trick: when an unrecognized voice says the wake phrase, EchoChamber reaches
out over `adb` and revokes the Google app's microphone permission — so *the phone itself*
stops listening for "Ok Google", not just this app. Say it yourself and it's granted right
back.

What you need:

- [`adb`](https://developer.android.com/tools/adb) installed and on your `PATH` (or
  installed via Android Studio / `winget install Google.PlatformTools`).
- An Android phone with **USB debugging** enabled (Settings → About phone → tap "Build
  number" 7 times → Developer options → USB debugging), plugged in and authorized
  (`adb devices` should list it as `device`, not `unauthorized`).

Then just tick **"Block 'Ok Google' for anyone else"** in the same panel and restart
capture. That's it — no app to install on the phone.

> Works on stock Android with the Google app. Samsung's Bixby wake word can't be blocked
> this way (Samsung hard-locks its mic permission) — see `echochamber/adb/hotword_core.py`
> for the details.

## What's actually happening

```
 mic ──► RingBuffer ──► WindowChunker ──► consumer thread ──► VoiceGateSink
                                                                    │
                                             phrase matched? ──► speaker verified (CAMPPlus)?
                                                                    │
                                                     yes ──────┬────┴──── no
                                                       (unblock, ok)  (block via adb)
```

- Everything upstream of the wake-phrase gate is a generic low-latency audio pipeline:
  sample-exact windows, a lock-free ring buffer, a bounded queue so a slow consumer can
  never stall the mic thread.
- The gate only starts caring once Vosk hears "ok google"/"hey google" — matched on whole
  words, so "look google it" doesn't trigger it.
- CAMPPlus (via ONNX Runtime QNN) turns that clip into a voice embedding and compares it
  against everyone enrolled.
- The result decides one thing: call `adb shell pm grant/revoke` on the Google app's
  `RECORD_AUDIO` permission. A crash-safety timer auto-restores it after 20 minutes no
  matter what, so a killed process can never leave your Assistant permanently blocked.

Every stage is optional and off by default — turn on only what you want, and the pipeline
degrades gracefully (a missing model, a disconnected phone, no adb found) instead of
crashing.

## Layout

| Path | What lives there |
|---|---|
| `echochamber/config.py`, `echochamber/audio/` | Core capture pipeline: ring buffer, windowing, sinks |
| `echochamber/voicegate/` | Wake-phrase detection (Vosk) and the snippet gate |
| `echochamber/speakerid/` | Speaker enrollment + verification (CAMPPlus / ONNX QNN) |
| `echochamber/adb/` | The ADB hotword-blocking trigger |
| `echochamber/ui/` | The GUI — `controller.py` holds all logic, panels stay dumb |
| `scripts/` | One-time setup: model downloads, venv builds, CLI enrollment |
| `docs/architecture.md` | Deep design rationale, latency model, and every hard-won gotcha |

## Notes for developers

- **Platform**: developed on Windows x86-64, deployed to Windows ARM64. Every dependency
  with a compiled extension needs a verified `win_arm64` wheel — that's why Vosk and the
  CAMPPlus/torch stack each run in their own subprocess venv instead of in-process.
- **WASAPI won't resample for you.** A 16 kHz request on a 48 kHz device fails unless the
  stream opens with `WasapiSettings(auto_convert=True)` — already handled, just noted here
  because it bites anyone extending the audio source.
- **Everything is optional and additive.** With no models installed at all, the app is
  still a fully working low-latency audio capture tool with live meters — the wake-phrase
  gate, speaker ID, and ADB trigger are three independent layers stacked on top.
- Prefer a terminal to a GUI? `python scripts/enroll_speaker.py --help` covers enrolling,
  listing, and removing speakers from the command line, `--wav` included.
- Full design rationale and the WebSocket event-notification feature (push a `detected`/
  `snippet` event to any listener) live in `docs/architecture.md`.
- Run the test suite (no microphone or phone required — everything is mocked/replayed):

  ```bash
  pytest
  ```
