import argparse
import asyncio
import csv
import json
import logging
import threading
import time
import datetime
import os
import webbrowser
from collections import deque
from dataclasses import dataclass
import numpy as np
import websockets
from typing import Optional

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    HAS_NIDAQMX = True
except ImportError:
    HAS_NIDAQMX = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION (OPEN FOR TUNING) ---
SAMPLE_RATE = 20000
# Chunk size dictates the latency. 
# 100 samples / 20000 Hz = 5 ms latency.
CHUNK_SIZE = 100  
VIS_DOWNSAMPLE_FACTOR = 100
UI_UPDATE_INTERVAL = 0.1 # Broadcast to UI every 100ms (10 FPS)
AI_CHANNELS = ["Dev1/ai0", "Dev1/ai1"]
AO_CHANNEL = ["Dev1/ao0"]
WS_PORT = 8765

# --- ESN CLOSED-LOOP CONFIG ---
ESN_ARTIFACT = os.path.join(os.path.dirname(__file__), "esn_artifact.pkl")
CTX_CHANNEL_INDEX = 1  # ai1 = EC/CTX (matches MockDAQManager ordering)
DEFAULT_STIM_GAIN = 1.0
DEFAULT_STIM_MODE = "passthrough"  # off | passthrough | threshold_pulse

try:
    from esn import load_artifact

    _ESN_STREAMER = load_artifact(ESN_ARTIFACT)
except Exception as _e:
    _ESN_STREAMER = None
    logger.warning(f"ESN artifact not loaded ({_e}). Closed-loop ESN will output zeros.")

@dataclass
class GlobalState:
    is_running: bool = True
    is_recording: bool = False
    mode: str = "control"  # 'control' or 'closed-loop'
    stim_mode: str = DEFAULT_STIM_MODE
    stim_gain: float = DEFAULT_STIM_GAIN

class DataLogger:
    """ Handles asynchronous dumping of recorded data to disk. """
    def __init__(self, file_prefix="recording"):
        self.file_prefix = file_prefix
        self.queue = deque()
        self.running = True
        self.file_handle = None
        self.writer = None
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()
        
    def start_recording(self):
        dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dt_str}_{self.file_prefix}.csv"
        self.file_handle = open(filename, mode='a', newline='')
        self.writer = csv.writer(self.file_handle)
        self.writer.writerow(["Time_s", "AI0", "AI1", "AO0", "Mode"])
        
    def stop_recording(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            self.writer = None

    def _writer_loop(self):
        while self.running:
            if len(self.queue) > 0:
                data_chunk = self.queue.popleft()
                if self.writer:
                    self.writer.writerows(data_chunk.T)
            else:
                time.sleep(0.01)

    def log_chunk(self, data_stacked):
        """ data_stacked is a numpy array of shape (4, CHUNK_SIZE) """
        self.queue.append(data_stacked)

    def stop(self):
        self.running = False
        self.stop_recording()
        self.thread.join()

def RCalgorithm(data_chunk):
    """
    data_chunk: shape (2, CHUNK_SIZE)
    Returns: stimulation_array of shape (1, CHUNK_SIZE)
    """
    if _ESN_STREAMER is None:
        return np.zeros((1, data_chunk.shape[1]))

    try:
        return _ESN_STREAMER.process_chunk(data_chunk, ctx_index=CTX_CHANNEL_INDEX)
    except Exception as e:
        logger.error(f"ESN processing error: {e}")
        return np.zeros((1, data_chunk.shape[1]))


class BaseDAQManager:
    """ Base class for DAQ operations (Mock or Real) """
    def __init__(self, state: GlobalState, logger: DataLogger, ws_queue: asyncio.Queue, loop):
        self.state = state
        self.logger = logger
        self.ws_queue = ws_queue
        self.loop = loop
        self.downsample_factor = VIS_DOWNSAMPLE_FACTOR  # For visualisation, no need to send 20kHz to UI
        self.total_samples = 0
        self.ui_buffer_ai = []
        self.ui_buffer_ao = []
        self.last_ui_push = time.time()

    def process_and_route(self, ai_data):
        """ Core algorithm application and routing """
        chunk_size = ai_data.shape[1]
        
        # 1. Process data for stimulation
        if self.state.mode == "closed-loop":
            ao_data = RCalgorithm(ai_data)
        else:
            ao_data = np.zeros((1, chunk_size))

        # 2. Log data if gap-free recording is enabled
        if self.state.is_recording:
            # Generate the time vector for this chunk
            time_vector = np.arange(self.total_samples, self.total_samples + chunk_size) / SAMPLE_RATE
            time_vector = time_vector.reshape(1, -1)
            
            # Numeric digital track for Mode (1.0 = closed-loop, 0.0 = control)
            mode_val = 1.0 if self.state.mode == 'closed-loop' else 0.0
            mode_vector = np.full((1, chunk_size), mode_val)
            
            # Stack all channels for disk
            all_data = np.vstack((time_vector, ai_data, ao_data, mode_vector))
            self.logger.log_chunk(all_data)
            
        self.total_samples += chunk_size

        # 3. Buffer and push to UI websocket queue asynchronously to prevent flooding
        self.ui_buffer_ai.append(ai_data)
        self.ui_buffer_ao.append(ao_data)

        if time.time() - self.last_ui_push >= UI_UPDATE_INTERVAL:
            try:
                combined_ai = np.hstack(self.ui_buffer_ai)
                combined_ao = np.hstack(self.ui_buffer_ao)
                
                downsampled_ai = combined_ai[:, ::self.downsample_factor].tolist()
                downsampled_ao = combined_ao[:, ::self.downsample_factor].tolist()
                
                packet = {
                    "ai": downsampled_ai,
                    "ao": downsampled_ao,
                    "mode": self.state.mode,
                    "is_recording": self.state.is_recording,
                    "stim_mode": getattr(self.state, "stim_mode", DEFAULT_STIM_MODE),
                    "stim_gain": getattr(self.state, "stim_gain", DEFAULT_STIM_GAIN),
                    "fs": SAMPLE_RATE / self.downsample_factor
                }
                
                self.loop.call_soon_threadsafe(self.ws_queue.put_nowait, json.dumps(packet))
            except Exception as e:
                pass  # Queue full or loop closed
            finally:
                self.ui_buffer_ai.clear()
                self.ui_buffer_ao.clear()
                self.last_ui_push = time.time()
            
        return ao_data

class RealDAQManager(BaseDAQManager):
    def run(self):
        logger.info("Initializing NI DAQ Tasks...")
        if not HAS_NIDAQMX:
            logger.error("nidaqmx is not installed or device drivers missing!")
            return

        with nidaqmx.Task() as read_task, nidaqmx.Task() as write_task:
            for ai in AI_CHANNELS:
                read_task.ai_channels.add_ai_voltage_chan(ai)
            for ao in AO_CHANNEL:
                write_task.ao_channels.add_ao_voltage_chan(ao)
             # AI is the timing master.
            read_task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=CHUNK_SIZE,
            )

            # AO uses the exact same hardware sample clock as AI.
            ai_sample_clock = read_task.timing.samp_clk_term

            write_task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                source=ai_sample_clock,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=CHUNK_SIZE,
            )
            # for task in (read_task, write_task):
            #     task.timing.cfg_samp_clk_timing(
            #         rate=SAMPLE_RATE, 
            #         sample_mode=AcquisitionType.CONTINUOUS,
            #         samps_per_chan=CHUNK_SIZE
            #     )

            # Sync Write task to Read task trigger
            write_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                read_task.triggers.start_trigger.term
            )
            
            # Initial zero write to prime the buffer
            initial_zeros = np.zeros((1, CHUNK_SIZE))
            write_task.write(initial_zeros.squeeze(), auto_start=False)
            
            write_task.start()
            read_task.start()

            logger.info("Hardware Acquisition started.")
            while self.state.is_running:
                # This call blocks until CHUNK_SIZE samples are available
                ai_data = np.array(read_task.read(number_of_samples_per_channel=CHUNK_SIZE))
                ao_data = self.process_and_route(ai_data)
                write_task.write(ao_data.squeeze(), auto_start=False)

class MockDAQManager(BaseDAQManager):
    """ DEBUG TOOL: Simulates AI channels using noise, ignoring hardware """
    def run(self):
        logger.info("Running MOCK DAQ...")
        while self.state.is_running:
            # Simulate wait time to acquire chunk
            time.sleep(CHUNK_SIZE / SAMPLE_RATE)
            noise_ca3 = np.random.normal(0, 0.5, CHUNK_SIZE)
            noise_ec = np.random.normal(0, 0.5, CHUNK_SIZE)
            
            if self.state.mode != 'closed-loop' and np.random.rand() < 0.05:
                noise_ca3 += np.sin(np.linspace(0, 50, CHUNK_SIZE)) * 5
                 
            ai_data = np.vstack((noise_ca3, noise_ec))
                
            self.process_and_route(ai_data)


# --- WEBSOCKET SERVER ---
async def websocket_handler(websocket, state: GlobalState, ws_queue: asyncio.Queue, data_logger: DataLogger):
    """ Handles UI Connections. Two parallel tasks: receiving commands, pushing data. """
    logger.info("UI Client connected.")
    
    async def rx():
        async for message in websocket:
            try:
                cmd = json.loads(message)
                if 'command' in cmd:
                    if cmd['command'] == 'start_recording':
                        if not state.is_recording:
                            data_logger.start_recording()
                            state.is_recording = True
                            logger.info("Gap-free recording STARTED")
                    elif cmd['command'] == 'stop_recording':
                        if state.is_recording:
                            state.is_recording = False
                            data_logger.stop_recording()
                            logger.info("Gap-free recording STOPPED")
                    elif cmd['command'] == 'set_mode':
                        state.mode = cmd.get('mode', 'control')
                        logger.info(f"Mode changed to: {state.mode}")
                        if state.mode == "control" and _ESN_STREAMER is not None:
                            _ESN_STREAMER.reset()
                    elif cmd["command"] == "set_stim":
                        # optional fields: stim_mode, stim_gain
                        if "stim_mode" in cmd:
                            state.stim_mode = cmd["stim_mode"]
                        if "stim_gain" in cmd:
                            state.stim_gain = float(cmd["stim_gain"])
                        if _ESN_STREAMER is not None:
                            _ESN_STREAMER.configure(stim_mode=state.stim_mode, stim_gain=state.stim_gain)
                        logger.info(f"Stim updated: mode={state.stim_mode}, gain={state.stim_gain}")
            except Exception as e:
                logger.error(f"WebSocket RX Error: {e}")

    async def tx():
        while state.is_running:
            try:
                # Wait for data from the DAQ manager
                data_packet = await ws_queue.get()
                await websocket.send(data_packet)
            except websockets.exceptions.ConnectionClosed:
                break
    
    await asyncio.gather(rx(), tx())

async def main():
    parser = argparse.ArgumentParser(description="Closed-Loop LFP System")
    parser.add_argument('--mock', action='store_true', help="DEBUG TOOL: Run without hardware using mocked random data.")
    args = parser.parse_args()

    state = GlobalState()
    ws_queue = asyncio.Queue()
    data_logger = DataLogger()
    loop = asyncio.get_running_loop()

    # Start DAQ in a separate system thread to avoid blocking the asyncio event loop
    if args.mock:
        daq_manager = MockDAQManager(state, data_logger, ws_queue, loop)
    else:
        daq_manager = RealDAQManager(state, data_logger, ws_queue, loop)
        
    daq_thread = threading.Thread(target=daq_manager.run, daemon=True)
    daq_thread.start()

    logger.info(f"Starting websocket server on port {WS_PORT}")
    ws_server = await websockets.serve(
        lambda ws: websocket_handler(ws, state, ws_queue, data_logger),
        "0.0.0.0", 
        WS_PORT
    )

    # Open the UI in the default browser automatically
    ui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "index.html"))
    logger.info(f"Opening Echo Chamber UI: {ui_path}")
    webbrowser.open(f"file://{ui_path}")

    try:
        # Keep main loop alive
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        state.is_running = False
        data_logger.stop()
        daq_thread.join(timeout=2.0)
        ws_server.close()
        await ws_server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
