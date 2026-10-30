from __future__ import annotations

import asyncio
import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import echoChamber as app


class ShutdownTransactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_shutdown_does_not_stop_recorder_before_owner_drain(self):
        class Recorder:
            stop_calls = 0

            def stop_recording(self):
                self.stop_calls += 1

        class WebSocket:
            def __init__(self):
                self.sent = []
                self._messages = iter([json.dumps({"command": "shutdown"})])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._messages)
                except StopIteration:
                    raise StopAsyncIteration

            async def send(self, message):
                self.sent.append(json.loads(message))

        state = app.RuntimeState(
            esn_ready=False, start_paused=False, electrode_labels=("CA3", "EC"), ctx_index=1,
            pulse_threshold_std=3.0, pulse_window_sec=10.0,
            controller_ready={"esn": False, "rule": True, "pid": True},
        )
        state.set_recording(True)
        recorder = Recorder()
        hub = app.UiHub(8)
        shutdown = asyncio.Event()
        websocket = WebSocket()
        await app.websocket_handler(websocket, state, recorder, hub, shutdown, app.AppConfig())

        self.assertTrue(shutdown.is_set())
        self.assertFalse(state.snapshot().recording)
        self.assertEqual(recorder.stop_calls, 0)
        self.assertEqual(websocket.sent[0]["command"], "shutdown")
        self.assertTrue(websocket.sent[0]["ok"])

    async def _run_main_with_blocked_owner(self, *, producer_release_delay_s: float) -> dict[str, object]:
        events: list[str] = []
        owner_release = threading.Event()
        owner_started = threading.Event()
        join_waiting = threading.Event()
        owner_join_ticks = [0]
        real_to_thread = asyncio.to_thread
        sample = Path(app.__file__).resolve().parent / "sample" / "test_AI0_CA3_AI1_CTX.h5"
        config = app.AppConfig(
            controller="rule", mock_replay=sample, record_dir=sample.parent,
            open_browser=False, start_paused=False,
        )

        class Parser:
            def parse_args(self):
                return SimpleNamespace(run_seconds=0.01, mock_overload_ms=0.0)

        class Esn:
            ready = False
            production_eligible = False
            error = None

        class Recorder:
            def __init__(self, *_args, **_kwargs):
                self.closed = False

            def close(self):
                self.closed = True
                events.append("recorder-close")

        class Core:
            def __init__(self, _config, _state, _esn, recorder, _hub, _loop, _telemetry):
                self.recorder = recorder

            def drain_ao_notifications(self):
                self.assert_recorder_open()
                events.append("core-drain")

            def assert_recorder_open(self):
                if self.recorder.closed:
                    raise AssertionError("Core notification drain ran after recorder close")

            def close_ui(self):
                events.append("core-ui-close")

        class Daq:
            def __init__(self, *_args, **_kwargs):
                self.request_count = 0

            def run(self):
                events.append("producer-start")
                owner_started.set()
                owner_release.wait()
                events.append("producer-return")

            def request_safe_zero(self):
                self.request_count += 1
                events.append("source-stop-request")
                if self.request_count == 1:
                    threading.Timer(producer_release_delay_s, owner_release.set).start()

        class Watchdog:
            def __init__(self, *_args, **_kwargs):
                pass

            def start(self):
                events.append("watchdog-start")

        class Server:
            sockets = ()

            def close(self):
                events.append("websocket-server-close")

            async def wait_closed(self):
                events.append("websocket-server-closed")

        async def serve(*_args, **_kwargs):
            return Server()

        async def controlled_to_thread(function, *args, **kwargs):
            if getattr(function, "__name__", "") == "join" and args == (8.0,):
                join_waiting.set()
                args = (0.005,)
            return await real_to_thread(function, *args, **kwargs)

        def heartbeat_task():
            async def beat():
                while not events or "recorder-close" not in events:
                    if join_waiting.is_set() and owner_started.is_set() and not owner_release.is_set():
                        owner_join_ticks[0] += 1
                    await asyncio.sleep(0.001)
            return asyncio.create_task(beat())

        args = Parser()
        with (
            patch.object(app, "build_parser", return_value=args),
            patch.object(app, "config_from_args", return_value=config),
            patch.object(app, "EsnRuntime", return_value=Esn()),
            patch.object(app, "H5Recorder", Recorder),
            patch.object(app, "AsyncUiProcessingCore", Core),
            patch.object(app, "MockDaq", Daq),
            patch.object(app, "Watchdog", Watchdog),
            patch.object(app.websockets, "serve", serve),
            patch.object(app.asyncio, "to_thread", controlled_to_thread),
        ):
            heartbeat = heartbeat_task()
            result = await app.main()
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

        self.assertIn("producer-return", events)
        self.assertLess(events.index("source-stop-request"), events.index("producer-return"))
        self.assertLess(events.index("producer-return"), events.index("core-drain"))
        self.assertLess(events.index("core-drain"), events.index("recorder-close"))
        return {"events": events, "result": result, "owner_join_ticks": owner_join_ticks[0]}

    async def test_main_waits_for_blocked_owner_then_drains_before_recorder_close(self):
        observed = await self._run_main_with_blocked_owner(producer_release_delay_s=0.08)
        events = observed["events"]
        self.assertGreaterEqual(events.count("source-stop-request"), 2)
        self.assertEqual(observed["result"], 1)  # bounded owner join timed out and faulted
        self.assertGreater(observed["owner_join_ticks"], 0)

    async def test_main_normal_owner_shutdown_drains_before_recorder_close(self):
        observed = await self._run_main_with_blocked_owner(producer_release_delay_s=0.001)
        events = observed["events"]
        self.assertEqual(events.count("source-stop-request"), 1)
        self.assertEqual(observed["result"], 0)


if __name__ == "__main__":
    unittest.main()
