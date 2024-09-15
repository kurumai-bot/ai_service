from datetime import datetime, timezone
from multiprocessing.connection import Listener
from queue import Empty, Queue
import traceback
from typing import Any, Dict
from uuid import uuid4

import orjson
import torch

from ai import Pipeline
from constants import AUTH_KEY, HF_API_KEY, HOST, LOGGER, OPENAI_API_KEY, PORT
from utils import Cache


cache = Cache(LOGGER.getChild("cache"))
send_queue = Queue()


def main():
    listener = Listener((HOST, PORT), authkey=AUTH_KEY)
    connection = listener.accept()
    LOGGER.info("Connected to backend")

    while True:
        try:
            try:
                # Send items in send queue
                try:
                    for _ in range(1000):
                        item = send_queue.get_nowait()
                        connection.send_bytes(item)
                except Empty:
                    pass

                # Read incoming bytes
                if not connection.poll(0.01):
                    continue

                payload = connection.recv_bytes()
            except (EOFError, OSError):
                LOGGER.warning("Backend disconnected from socket. Attempting reconnect...")
                connection = listener.accept()
                LOGGER.info("Reconnected to backend")
                continue

            if payload[0] == 3:
                sep = payload.index(0xff)
                recv_voice_data(payload[1:sep].decode(), payload[sep + 1:])
            else:
                event = orjson.loads(payload)

                if event["op"] == 1:
                    set_preset(event["id"], event["preset"])
                elif event["op"] == 2:
                    remove_preset(event["id"])
                elif event["op"] == 4:
                    recv_text_data(event["id"], event["data"])
                else:
                    LOGGER.warning("Received unrecognized event in listener loop: %s", event)
        except Exception: # pylint: disable=broad-exception-caught
            LOGGER.warning("Exception in listener loop:\n%s", traceback.format_exc())


def set_preset(id: str, preset: Dict[str, Any]):
    LOGGER.debug("Received set preset for user: %s", id)
    cached_pipeline = cache.get(id)
    if cached_pipeline is None or cached_pipeline[1][0] != preset["id"]:
        # TODO: Make pipeline creation multithreaded
        pipeline = cache.get(id) or Pipeline(
            "openai/whisper-base.en",
            preset["tts_model_name"],
            preset["tts_speaker_name"],
            preset["text_gen_model_name"],
            preset["text_gen_starting_context"],
            pipeline_callback,
            openai_api_key=OPENAI_API_KEY,
            hf_api_key=HF_API_KEY,
            logger=LOGGER.getChild("pipeline"),
            low_cpu_mem_usage=True,
            device_map="cuda:0",
            torch_dtype=torch.float16
        )
    else:
        pipeline = cached_pipeline[1][1]
    pipeline.start()
    cache.add(id, pipeline)

def remove_preset(id: str):
    LOGGER.debug("Removing preset for user: %s", id)
    cache.remove(id)

# TODO: cache entry may expire while user is still connected. handle this better
# TODO: pipeline may not exist sometimes, add some way to handle that
# TODO: probably rethink error codes/standardize them at some point
def recv_voice_data(id: str, data: bytes):
    if cache.get(id) is None:
        send_error(id, 0)
        return
    cache.get(id)[1].process_input(data, datetime.now(timezone.utc), id)

def recv_text_data(id: str, data: str):
    # TODO: Error handling
    if cache.get(id) is None:
        send_error(id, 0)
        return
    LOGGER.debug("Received text for user: %s", id)
    cache.get(id)[1].process_input(data, datetime.now(timezone.utc), id)

def pipeline_callback(event: str, timestamp: datetime, result: Any, id: str):
    match event:
        case "start":
            opcode = 5
        case "finish_asr":
            opcode = 6
        case "finish_gen":
            opcode = 7
            wav = result.pop("wav")
            result["wav_id"] = uuid4()
        case "finish":
            opcode = 9
        case _:
            raise ValueError("Unrecognized event in pipeline callback: " + event)

    send_queue.put(orjson.dumps({
        "op": opcode,
        "id": id,
        "timestamp": timestamp,
        "data": result
    }))

    # Send wav separately to save on serialization time
    if opcode == 7:
        id_bytes = id.encode()
        payload = bytearray(1 + len(id_bytes) + 1 + 16 + len(wav))
        pos = 0
        payload[pos] = 8

        pos += 1
        payload[pos:pos + len(id_bytes)] = id_bytes

        pos += len(id_bytes)
        payload[pos] = 0xff

        pos += 1
        payload[pos:pos + 16] = result["wav_id"].bytes

        pos += 16
        payload[pos:] = wav.tobytes()
        send_queue.put(bytes(payload))


def send_error(id: str, error_code: int) -> None:
    LOGGER.info("Sending error to server: %i, %s", error_code, id)
    send_queue.put(orjson.dumps({
        "op": 0,
        "id": id,
        "timestamp": datetime.now(timezone.utc),
        "data": error_code
    }))


if __name__ == "__main__":
    LOGGER.info("Starting listener server")
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down...")
        cache.close()
