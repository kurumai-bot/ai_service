from datetime import datetime, timezone
from multiprocessing.connection import Listener
from queue import Empty, Queue
import traceback
from typing import Any, Dict
from uuid import uuid4

import orjson

from ai import Pipeline
from constants import AUTH_KEY, HOST, LOGGER, OPENAI_API_KEY, PORT
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
                if not connection.poll(1):
                    continue

                payload = connection.recv_bytes()
            except (EOFError, OSError):
                LOGGER.warning("Backend disconnected from socket. Attempting reconnect...")
                connection = listener.accept()
                LOGGER.info("Reconnected to backend")

            if payload[0] == 3:
                pass
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
        )
    else:
        pipeline = cached_pipeline[1][1]
    pipeline.start()
    cache.add(id, (preset["id"], pipeline))

def remove_preset(id: str):
    cache.remove(id)

def recv_voice_data(id: str, data: bytes):
    cache.get(id)[1][1].process_input(data, datetime.now(timezone.utc), id)

def recv_text_data(id: str, data: str):
    # TODO: Error handling
    cache.get(id)[1][1].process_input(data, datetime.now(timezone.utc), id)

def pipeline_callback(event: str, timestamp: datetime, result: Any, id: str):
    if event == "start":
        opcode = 5
    elif event == "finish_asr":
        opcode = 6
    elif event == "finish_gen":
        opcode = 7
        wav = result.pop("wav")
        result["wav_id"] = uuid4()
    elif event == "finish":
        opcode = 9
    else:
        raise ValueError("Unrecognized event in pipeline callback: " + event)

    send_queue.put(orjson.dumps({
        "op": opcode,
        "id": id,
        "timestamp": timestamp,
        "data": result
    }))

    # Send wav separately to save on serialization time
    if opcode == 7:
        payload = bytearray(1 + len(id) + 1 + 32 + len(wav))
        pos = 0
        payload[pos] = 8

        pos += 1
        payload[pos:pos + len(id)] = bytes(id, encoding="ascii")

        pos += len(id)
        payload[pos] = 0xFF

        pos += 1
        payload[pos:pos + 32] = result["wav_id"].bytes

        pos += 32
        payload[pos:] = wav.tobytes()
        send_queue.put(bytes(payload))


if __name__ == "__main__":
    LOGGER.info("Starting listener server")
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down...")
        cache.close()
