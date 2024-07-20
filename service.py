from datetime import datetime, timezone
from multiprocessing.connection import Listener
import traceback
from typing import Any, Dict
from uuid import UUID

import orjson

from ai import Pipeline
from constants import AUTH_KEY, HOST, LOGGER, OPENAI_API_KEY, PORT
from utils import Cache


cache = Cache(LOGGER.getChild("cache"))


def main():
    listener = Listener((HOST, PORT), authkey=AUTH_KEY)
    connection = listener.accept()
    while True:
        try:
            try:
                payload = connection.recv_bytes()
            except (EOFError, OSError):
                LOGGER.warning("Backend disconnected from socket. Attempting reconnect...")
                connection = listener.accept()

            if payload[0] == 3:
                pass
            else:
                event = orjson.loads(payload)
                event["id"] = UUID(event["id"])

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
        except KeyboardInterrupt:
            cache.close()
            return


def set_preset(user_id: UUID, preset: Dict[str, Any]):
    cached_pipeline = cache.get(str(user_id))
    if cached_pipeline is None or cached_pipeline[1][0] != preset["id"]:
        # TODO: Make pipeline creation multithreaded
        pipeline = cache.get(str(user_id)) or Pipeline(
            "openai/whisper-base.en",
            preset["tts_model_name"],
            preset["tts_speaker_name"],
            preset["text_gen_model_name"],
            preset["text_gen_starting_context"],
            callback=lambda *args, **kwargs: print(args, kwargs),
            openai_api_key=OPENAI_API_KEY,
        )
    else:
        pipeline = cached_pipeline
    pipeline.start()
    cache.add(str(user_id), (preset["id"], pipeline))

def remove_preset(user_id: UUID):
    cache.remove(str(user_id))

def recv_voice_data(user_id: UUID, data: bytes):
    pass

def recv_text_data(user_id: UUID, data: str):
    # TODO: Error handling
    cache.get(str(user_id))[1][1].process_input(data, datetime.now(timezone.utc))

if __name__ == "__main__":
    main()
