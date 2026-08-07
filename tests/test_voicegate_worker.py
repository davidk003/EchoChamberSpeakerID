"""Tests for echochamber.voicegate.worker, driven in-process against BytesIO.

:func:`~echochamber.voicegate.worker.run` takes its two streams as arguments
rather than reaching for ``sys.stdin``/``sys.stdout`` precisely so it can be
driven like this: **no subprocess is launched anywhere in this file**, and no
model is loaded.  The whole frame loop is exercised by handing ``run`` an input
:class:`io.BytesIO` pre-loaded with frames and reading the output one back with
:func:`~echochamber.voicegate.protocol.read_frame`.

Model loading is the one thing that cannot happen here, so
``worker.load_vosk_recognizer`` is monkeypatched -- to a
:class:`~echochamber.voicegate.recognizer.ScriptedRecognizer` for the happy
paths, and to something that raises for the failure path.  Patching the name
*in the worker module* rather than in ``recognizer`` is deliberate: it is the
binding ``run`` actually calls.

The assertions are on decoded frames, not on byte counts.  ``run`` returning 0
while writing nothing would pass a smoke test; here the output stream is
decoded to a list of ``(kind, payload)`` pairs and compared exactly, which is
also how the READY handshake's contents get pinned down -- the parent blocks on
that frame, and its payload is a compatibility surface between two
interpreters.
"""

from __future__ import annotations

import io
import json
from typing import Any, Callable

import pytest

from echochamber.voicegate import worker
from echochamber.voicegate.protocol import (
    FrameKind,
    decode_json,
    encode_frame,
    encode_json,
    read_frame,
)
from echochamber.voicegate.recognizer import Recognition, ScriptedRecognizer


MODEL = "/models/vosk-model-small-en-us-0.15"
SR = 16_000


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def inbound(*frames: tuple[FrameKind, bytes] | FrameKind) -> io.BytesIO:
    """Build the worker's stdin from ``FrameKind`` or ``(kind, payload)`` items."""
    blob = b""
    for item in frames:
        if isinstance(item, FrameKind):
            blob += encode_frame(item)
        else:
            blob += encode_frame(item[0], item[1])
    return io.BytesIO(blob)


def outbound(stream: io.BytesIO) -> list[tuple[FrameKind, bytes]]:
    """Decode everything the worker wrote, in order."""
    stream.seek(0)
    frames: list[tuple[FrameKind, bytes]] = []
    while True:
        frame = read_frame(stream)
        if frame is None:
            return frames
        frames.append(frame)


def kinds(frames: list[tuple[FrameKind, bytes]]) -> list[FrameKind]:
    """Just the kinds, for asserting the shape of an exchange."""
    return [kind for kind, _ in frames]


def script(*items: tuple[int, str, bool]) -> list[tuple[int, Recognition]]:
    """A ScriptedRecognizer script from ``(offset, text, final)`` triples."""
    return [(o, Recognition(text=t, final=f)) for o, t, f in items]


def patch_loader(
    monkeypatch: pytest.MonkeyPatch,
    recognizer: Any = None,
    raises: BaseException | None = None,
) -> list[tuple[str, int, tuple[str, ...]]]:
    """Replace ``worker.load_vosk_recognizer``; return the list of calls made.

    Patching the name inside :mod:`echochamber.voicegate.worker` rather than in
    :mod:`echochamber.voicegate.recognizer` is what makes this work: ``run``
    resolved the function at import time into the worker module's globals.
    """
    calls: list[tuple[str, int, tuple[str, ...]]] = []

    def fake_load(model_path: str, sample_rate: int, phrases: tuple[str, ...] = ()):
        calls.append((model_path, sample_rate, tuple(phrases)))
        if raises is not None:
            raise raises
        return recognizer if recognizer is not None else ScriptedRecognizer()

    monkeypatch.setattr(worker, "load_vosk_recognizer", fake_load)
    return calls


def audio(n: int) -> tuple[FrameKind, bytes]:
    """An AUDIO frame carrying ``n`` bytes of PCM; only the length matters."""
    return (FrameKind.AUDIO, bytes(n))


class Boom(RuntimeError):
    """Sentinel raised by a deliberately broken model loader."""


# ==========================================================================
# module import hygiene
# ==========================================================================

class TestModuleImport:
    """Importing the worker must do nothing; everything lives inside main()."""

    def test_import_exposes_the_entry_points_without_running_them(self) -> None:
        """The test suite imports this module in-process; a side effect would
        take stdio away from pytest."""
        for name in ("build_parser", "main", "run", "set_binary_mode"):
            assert callable(getattr(worker, name)), f"{name} must be importable"
        assert worker.__all__ == ["build_parser", "main", "run", "set_binary_mode"]

    def test_importing_does_not_import_vosk(self) -> None:
        """The lazy import is the reason a checkout without Vosk still runs."""
        import sys

        assert "vosk" not in sys.modules, (
            "importing the worker must not pull in vosk; it is imported lazily "
            "inside load_vosk_recognizer"
        )


# ==========================================================================
# build_parser
# ==========================================================================

class TestBuildParser:
    """The argument surface is a contract with SubprocessRecognizer.command."""

    def test_model_is_required(self) -> None:
        """Without --model there is nothing to load, so argparse must refuse."""
        with pytest.raises(SystemExit) as excinfo:
            worker.build_parser().parse_args([])
        assert excinfo.value.code == 2, "argparse exits 2 on a usage error"

    def test_model_is_parsed(self) -> None:
        """--model lands on args.model verbatim."""
        args = worker.build_parser().parse_args(["--model", MODEL])
        assert args.model == MODEL

    def test_sample_rate_defaults_to_16000(self) -> None:
        """The default must match the capture rate the pipeline ships with."""
        args = worker.build_parser().parse_args(["--model", MODEL])
        assert args.sample_rate == 16_000
        assert isinstance(args.sample_rate, int)

    def test_sample_rate_is_parsed_as_an_int(self) -> None:
        """The value reaches Vosk as a number, not as the string '48000'."""
        args = worker.build_parser().parse_args(
            ["--model", MODEL, "--sample-rate", "48000"]
        )
        assert args.sample_rate == 48_000

    def test_phrase_defaults_to_none(self) -> None:
        """No --phrase means open vocabulary, which run() turns into ()."""
        args = worker.build_parser().parse_args(["--model", MODEL])
        assert args.phrases is None

    def test_phrase_is_repeatable_into_a_list(self) -> None:
        """SubprocessRecognizer.command emits one --phrase per phrase."""
        args = worker.build_parser().parse_args(
            [
                "--model", MODEL,
                "--phrase", "ok chamber",
                "--phrase", "hey chamber",
                "--phrase", "start recording",
            ]
        )
        assert args.phrases == ["ok chamber", "hey chamber", "start recording"]

    def test_a_single_phrase_is_still_a_list(self) -> None:
        """append always yields a list, so run() never sees a bare string."""
        args = worker.build_parser().parse_args(
            ["--model", MODEL, "--phrase", "ok chamber"]
        )
        assert args.phrases == ["ok chamber"]

    def test_log_level_defaults_to_none(self) -> None:
        """Logging is off unless asked for; stdout is the frame channel."""
        args = worker.build_parser().parse_args(["--model", MODEL])
        assert args.log_level is None

    def test_full_command_line_parses(self) -> None:
        """The exact argv SubprocessRecognizer builds must be accepted."""
        args = worker.build_parser().parse_args(
            ["--model", MODEL, "--sample-rate", "16000", "--phrase", "ok chamber"]
        )
        assert (args.model, args.sample_rate, args.phrases) == (
            MODEL, 16_000, ["ok chamber"]
        )


# ==========================================================================
# set_binary_mode
# ==========================================================================

class TestSetBinaryMode:
    """A no-op off Windows, and harmless on a stream with no real descriptor."""

    def test_returns_the_same_stream_object(self) -> None:
        """It is used inline where the stream is bound, so it must return it."""
        stream = io.BytesIO()
        assert worker.set_binary_mode(stream) is stream

    def test_does_not_raise_on_a_bytesio(self) -> None:
        """BytesIO.fileno() raises; the failure is swallowed, not propagated."""
        stream = io.BytesIO(b"payload")
        worker.set_binary_mode(stream)
        assert stream.read() == b"payload", "the stream must be left usable"

    def test_does_not_consume_or_reposition_the_stream(self) -> None:
        """Switching modes must not disturb a byte -- PCM is already in flight."""
        stream = io.BytesIO(b"abcdef")
        stream.seek(2)
        worker.set_binary_mode(stream)
        assert stream.tell() == 2


# ==========================================================================
# run() -- the frame loop
# ==========================================================================

class TestRunHandshake:
    """READY is the first thing on the wire; the parent blocks on it."""

    def test_ready_is_written_before_anything_else(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Audio sent before READY would be decoded by a model that does not exist."""
        patch_loader(monkeypatch, ScriptedRecognizer(script((1, "hi", True))))
        stdout = io.BytesIO()

        rc = worker.run(inbound(audio(8), FrameKind.SHUTDOWN), stdout, MODEL, SR)

        assert rc == 0
        assert kinds(outbound(stdout))[0] is FrameKind.READY

    def test_ready_payload_names_the_model_rate_and_phrases(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The handshake is a compatibility surface between two interpreters."""
        patch_loader(monkeypatch)
        stdout = io.BytesIO()

        worker.run(
            inbound(FrameKind.SHUTDOWN),
            stdout,
            MODEL,
            48_000,
            ("ok chamber", "hey chamber"),
        )

        kind, payload = outbound(stdout)[0]
        assert kind is FrameKind.READY
        assert decode_json(payload) == {
            "model": MODEL,
            "sample_rate": 48_000,
            "phrases": ["ok chamber", "hey chamber"],
        }

    def test_ready_payload_has_an_empty_phrase_list_for_open_vocabulary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tuple must serialise as a JSON array, not vanish."""
        patch_loader(monkeypatch)
        stdout = io.BytesIO()
        worker.run(inbound(FrameKind.SHUTDOWN), stdout, MODEL, SR)
        assert decode_json(outbound(stdout)[0][1])["phrases"] == []

    def test_the_loader_receives_the_arguments_run_was_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run() must not silently substitute defaults for what it was passed."""
        calls = patch_loader(monkeypatch)
        worker.run(inbound(FrameKind.SHUTDOWN), io.BytesIO(), MODEL, 8_000, ("go",))
        assert calls == [(MODEL, 8_000, ("go",))]


class TestRunHappyPath:
    """Audio in, RESULT frames out, clean exit code."""

    def test_audio_then_shutdown_returns_zero_with_the_scripted_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The headline test: what the recogniser emitted is what reaches the wire."""
        recognizer = ScriptedRecognizer(
            script((4, "hey chamber", True), (12, "start recording", True))
        )
        patch_loader(monkeypatch, recognizer)
        stdout = io.BytesIO()

        rc = worker.run(
            inbound(audio(4), audio(4), audio(4), FrameKind.SHUTDOWN),
            stdout,
            MODEL,
            SR,
        )

        assert rc == 0
        frames = outbound(stdout)
        assert kinds(frames) == [FrameKind.READY, FrameKind.RESULT, FrameKind.RESULT]
        assert [decode_json(p) for _, p in frames[1:]] == [
            {"text": "hey chamber", "final": True, "confidence": 0.0},
            {"text": "start recording", "final": True, "confidence": 0.0},
        ]

    def test_the_recognizer_receives_the_audio_payloads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The PCM handed to the decoder is the frame body, byte for byte."""
        recognizer = ScriptedRecognizer()
        patch_loader(monkeypatch, recognizer)

        worker.run(
            inbound(audio(160), audio(320), FrameKind.SHUTDOWN),
            io.BytesIO(),
            MODEL,
            SR,
        )
        assert recognizer.consumed == 480

    def test_several_results_from_one_buffer_become_several_frames(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One accept_pcm may settle several results; each gets its own frame."""
        patch_loader(
            monkeypatch,
            ScriptedRecognizer(script((1, "a", False), (2, "b", False), (3, "c", True))),
        )
        stdout = io.BytesIO()

        worker.run(inbound(audio(100), FrameKind.SHUTDOWN), stdout, MODEL, SR)

        frames = outbound(stdout)
        assert kinds(frames) == [FrameKind.READY] + [FrameKind.RESULT] * 3
        assert [decode_json(p)["text"] for _, p in frames[1:]] == ["a", "b", "c"]

    def test_a_partial_is_marked_final_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate only acts on finals, so the flag must survive the wire."""
        patch_loader(monkeypatch, ScriptedRecognizer(script((1, "hey cham", False))))
        stdout = io.BytesIO()

        worker.run(inbound(audio(4), FrameKind.SHUTDOWN), stdout, MODEL, SR)

        assert decode_json(outbound(stdout)[1][1]) == {
            "text": "hey cham", "final": False, "confidence": 0.0
        }

    def test_confidence_is_serialised_as_a_float(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A numpy or int confidence must reach the parent as a JSON number."""
        patch_loader(
            monkeypatch,
            ScriptedRecognizer([(1, Recognition("hi", True, 0.875))]),
        )
        stdout = io.BytesIO()

        worker.run(inbound(audio(4), FrameKind.SHUTDOWN), stdout, MODEL, SR)

        assert decode_json(outbound(stdout)[1][1])["confidence"] == pytest.approx(0.875)

    def test_silence_produces_only_the_ready_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Most buffers settle nothing; the worker must not chatter."""
        patch_loader(monkeypatch, ScriptedRecognizer())
        stdout = io.BytesIO()

        rc = worker.run(
            inbound(audio(3200), audio(3200), FrameKind.SHUTDOWN), stdout, MODEL, SR
        )

        assert rc == 0
        assert kinds(outbound(stdout)) == [FrameKind.READY]

    def test_the_recognizer_is_closed_on_the_way_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The finally clause must release the model however the loop ended."""
        recognizer = ScriptedRecognizer()
        patch_loader(monkeypatch, recognizer)

        worker.run(inbound(FrameKind.SHUTDOWN), io.BytesIO(), MODEL, SR)
        assert recognizer.closed is True

    def test_clean_eof_without_shutdown_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A parent that just closed our stdin exited normally, not abnormally."""
        recognizer = ScriptedRecognizer()
        patch_loader(monkeypatch, recognizer)
        stdout = io.BytesIO()

        rc = worker.run(inbound(audio(8), audio(8)), stdout, MODEL, SR)

        assert rc == 0, "an end of stream is a clean shutdown, not a failure"
        assert kinds(outbound(stdout)) == [FrameKind.READY]
        assert recognizer.closed is True

    def test_an_empty_stdin_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A parent that died before sending anything is still a clean EOF."""
        patch_loader(monkeypatch)
        stdout = io.BytesIO()
        assert worker.run(io.BytesIO(), stdout, MODEL, SR) == 0
        assert kinds(outbound(stdout)) == [FrameKind.READY]

    def test_frames_after_shutdown_are_not_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SHUTDOWN ends the loop immediately; trailing audio is not decoded."""
        recognizer = ScriptedRecognizer()
        patch_loader(monkeypatch, recognizer)

        worker.run(
            inbound(audio(10), FrameKind.SHUTDOWN, audio(9999)),
            io.BytesIO(),
            MODEL,
            SR,
        )
        assert recognizer.consumed == 10


class TestRunControlFrames:
    """RESET and the frames that travel the wrong way."""

    def test_reset_frame_resets_the_recognizer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A discontinuity in the capture must reach the decoder."""
        recognizer = ScriptedRecognizer()
        patch_loader(monkeypatch, recognizer)

        rc = worker.run(
            inbound(audio(4), FrameKind.RESET, audio(4), FrameKind.RESET,
                    FrameKind.SHUTDOWN),
            io.BytesIO(),
            MODEL,
            SR,
        )

        assert rc == 0
        assert recognizer.resets == 2, (
            f"two RESET frames must produce two reset() calls, got {recognizer.resets}"
        )

    def test_reset_produces_no_output_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RESET is fire-and-forget; the parent does not wait for an ack."""
        patch_loader(monkeypatch)
        stdout = io.BytesIO()
        worker.run(inbound(FrameKind.RESET, FrameKind.SHUTDOWN), stdout, MODEL, SR)
        assert kinds(outbound(stdout)) == [FrameKind.READY]

    @pytest.mark.parametrize(
        "kind", [FrameKind.READY, FrameKind.RESULT, FrameKind.ERROR],
        ids=lambda k: k.name,
    )
    def test_an_unexpected_inbound_kind_is_ignored_not_fatal(
        self, kind: FrameKind, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dropping an unknown control frame keeps the audio flowing.

        These three kinds travel worker-to-parent.  A future parent may send
        something this worker predates, and dying over it would take the whole
        recogniser down for a frame that could simply be skipped.
        """
        recognizer = ScriptedRecognizer(script((8, "still here", True)))
        patch_loader(monkeypatch, recognizer)
        stdout = io.BytesIO()

        rc = worker.run(
            inbound((kind, b"unexpected"), audio(8), FrameKind.SHUTDOWN),
            stdout,
            MODEL,
            SR,
        )

        assert rc == 0, f"an inbound {kind.name} frame must not be fatal"
        frames = outbound(stdout)
        assert kinds(frames) == [FrameKind.READY, FrameKind.RESULT], (
            "the loop must carry on and still decode the audio behind it"
        )
        assert decode_json(frames[1][1])["text"] == "still here"


class TestRunFailures:
    """Both ways run() can return 1, each reported as an ERROR frame."""

    def test_model_load_failure_returns_one_with_a_single_error_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dying silently would leave the parent to time out saying nothing useful."""
        patch_loader(monkeypatch, raises=Boom("model directory is a banana"))
        stdout = io.BytesIO()

        rc = worker.run(inbound(FrameKind.SHUTDOWN), stdout, MODEL, SR)

        assert rc == 1
        frames = outbound(stdout)
        assert kinds(frames) == [FrameKind.ERROR], (
            "a failed load must produce exactly one ERROR frame and no READY"
        )

    def test_the_error_frame_carries_the_exception_type_and_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'model failed to load' without a traceback is unactionable."""
        patch_loader(monkeypatch, raises=Boom("model directory is a banana"))
        stdout = io.BytesIO()

        worker.run(inbound(FrameKind.SHUTDOWN), stdout, MODEL, SR)

        message = outbound(stdout)[0][1].decode("utf-8")
        assert "Boom" in message, "the exception type must be named"
        assert "model directory is a banana" in message, "the message must survive"
        assert "Traceback (most recent call last)" in message, (
            "the traceback is the whole point of the ERROR frame"
        )

    def test_a_filenotfounderror_from_the_loader_is_reported_the_same_way(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real failure is a missing model directory, not a synthetic Boom."""
        patch_loader(
            monkeypatch,
            raises=FileNotFoundError(f"Vosk model directory not found: {MODEL!r}"),
        )
        stdout = io.BytesIO()

        assert worker.run(inbound(FrameKind.SHUTDOWN), stdout, MODEL, SR) == 1
        message = outbound(stdout)[0][1].decode("utf-8")
        assert "FileNotFoundError" in message and MODEL in message

    def test_the_input_stream_is_not_read_after_a_load_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """There is no recogniser to feed, so the loop must not start."""
        patch_loader(monkeypatch, raises=Boom("nope"))
        stdin = inbound(audio(8), FrameKind.SHUTDOWN)

        worker.run(stdin, io.BytesIO(), MODEL, SR)
        assert stdin.tell() == 0, "run() must not consume stdin when the load failed"

    def test_a_desynchronised_stream_returns_one_and_reports_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A length-prefixed protocol cannot resync, so it stops and says so."""
        patch_loader(monkeypatch)
        stdout = io.BytesIO()
        garbage = io.BytesIO(b"\x99\x00\x00\x00\x00 this is not a frame stream")

        rc = worker.run(garbage, stdout, MODEL, SR)

        assert rc == 1
        frames = outbound(stdout)
        assert kinds(frames) == [FrameKind.READY, FrameKind.ERROR]
        assert "unknown frame kind 153" in frames[1][1].decode("utf-8")

    def test_a_truncated_frame_returns_one_and_reports_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A parent killed mid-frame is a torn stream, not a clean EOF."""
        patch_loader(monkeypatch)
        stdout = io.BytesIO()
        torn = io.BytesIO(encode_frame(FrameKind.AUDIO, b"12345678")[:-3])

        assert worker.run(torn, stdout, MODEL, SR) == 1
        frames = outbound(stdout)
        assert kinds(frames) == [FrameKind.READY, FrameKind.ERROR]
        assert "AUDIO payload" in frames[1][1].decode("utf-8")

    def test_a_desynchronised_stream_still_closes_the_recognizer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The finally clause covers the failure path too."""
        recognizer = ScriptedRecognizer()
        patch_loader(monkeypatch, recognizer)

        worker.run(io.BytesIO(b"\x00\x00\x00\x00\x00"), io.BytesIO(), MODEL, SR)
        assert recognizer.closed is True

    def test_garbage_after_good_frames_reports_the_frames_it_managed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Results decoded before the desync are not thrown away."""
        patch_loader(monkeypatch, ScriptedRecognizer(script((4, "heard this", True))))
        stdout = io.BytesIO()
        stdin = io.BytesIO(encode_frame(FrameKind.AUDIO, b"1234") + b"\xfe\x00\x00\x00\x00")

        assert worker.run(stdin, stdout, MODEL, SR) == 1
        assert kinds(outbound(stdout)) == [
            FrameKind.READY, FrameKind.RESULT, FrameKind.ERROR
        ]


# ==========================================================================
# main()
# ==========================================================================

class TestMain:
    """The CLI wrapper: parse, switch stdio to binary, hand off to run()."""

    def test_help_exits_zero(self) -> None:
        """--help is a successful invocation, and must not need a model."""
        with pytest.raises(SystemExit) as excinfo:
            worker.main(["--help"])
        assert excinfo.value.code == 0

    def test_missing_model_exits_two(self) -> None:
        """A usage error must not be confused with a runtime failure."""
        with pytest.raises(SystemExit) as excinfo:
            worker.main([])
        assert excinfo.value.code == 2

    def test_main_forwards_parsed_arguments_to_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--phrase repetition must arrive at run() as a tuple, not a list."""
        seen: list[tuple[Any, ...]] = []

        def fake_run(stdin, stdout, model_path, sample_rate, phrases=()):
            seen.append((model_path, sample_rate, phrases))
            return 0

        monkeypatch.setattr(worker, "run", fake_run)
        monkeypatch.setattr(worker.sys, "stdin", _FakeStdio())
        monkeypatch.setattr(worker.sys, "stdout", _FakeStdio())

        rc = worker.main(
            ["--model", MODEL, "--sample-rate", "8000", "--phrase", "a", "--phrase", "b"]
        )

        assert rc == 0
        assert seen == [(MODEL, 8_000, ("a", "b"))]

    def test_main_turns_no_phrases_into_an_empty_tuple(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """args.phrases is None by default; run() must never see None."""
        seen: list[Any] = []
        monkeypatch.setattr(
            worker, "run", lambda *a, **kw: (seen.append(a[-1]), 0)[1]
        )
        monkeypatch.setattr(worker.sys, "stdin", _FakeStdio())
        monkeypatch.setattr(worker.sys, "stdout", _FakeStdio())

        worker.main(["--model", MODEL])
        assert seen == [()]

    def test_main_returns_the_exit_code_run_produced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The process exit code is run()'s, so a failed load is visible outside."""
        monkeypatch.setattr(worker, "run", lambda *a, **kw: 1)
        monkeypatch.setattr(worker.sys, "stdin", _FakeStdio())
        monkeypatch.setattr(worker.sys, "stdout", _FakeStdio())
        assert worker.main(["--model", MODEL]) == 1


class _FakeStdio:
    """A stand-in for ``sys.stdin``/``sys.stdout`` exposing only ``.buffer``."""

    def __init__(self) -> None:
        self.buffer = io.BytesIO()


# ==========================================================================
# _encode_result -- the RESULT payload shape
# ==========================================================================

class TestEncodeResult:
    """The RESULT payload is what the parent's _decode_result reads back."""

    def test_encodes_the_three_documented_fields(self) -> None:
        """text/final/confidence, and nothing else the parent must guess at."""
        payload = worker._encode_result(Recognition("hey chamber", True, 0.5))
        assert json.loads(payload.decode("utf-8")) == {
            "text": "hey chamber", "final": True, "confidence": 0.5
        }

    def test_coerces_final_to_a_bool_and_confidence_to_a_float(self) -> None:
        """JSON has no numpy types; coercion happens before serialisation."""
        payload = worker._encode_result(Recognition("x", final=1, confidence=1))  # type: ignore[arg-type]
        parsed = json.loads(payload.decode("utf-8"))
        assert parsed["final"] is True
        assert isinstance(parsed["confidence"], float)

    def test_round_trips_through_encode_json(self) -> None:
        """The payload is valid JSON the protocol helpers can decode."""
        payload = worker._encode_result(Recognition("", False, 0.0))
        assert decode_json(payload) == {"text": "", "final": False, "confidence": 0.0}
        assert payload == encode_json({"text": "", "final": False, "confidence": 0.0})
