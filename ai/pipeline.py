from datetime import datetime
import logging
from queue import Empty, Full, PriorityQueue
import re
from threading import Thread
import traceback
from typing import Any, Callable, Tuple

import numpy as np

from ai.asr_processor import ASRProcessor
from ai.text_gen_processor import TextGenProcessor
from ai.tts_processor import TTSProcessor
from db import ModelPreset
from utils import CircularBuffer


sentence_end_regex = re.compile(r"[.?!][.?!\s]+")


class Pipeline:
    def __init__(
        self,
        model_preset: ModelPreset,
        callback: Callable[[str, datetime, Any, Any], Any],
        **kwargs
    ) -> None:
        #TODO: Don't create new models for every connection

        # Start logger
        self.logger = kwargs.pop("logger", None) or logging.getLogger("pipeline")

        self.asr_buffer_length = kwargs.pop("asr_buffer_size", 15)
        self.max_queue_audio_bytes = kwargs.pop("max_queue_audio_bytes", 16_000 * 100)
        self.max_queue_gen_chars = kwargs.pop("max_queue_gen_chars", 3000)\

        asr_logger = kwargs.pop("asr_logger")
        tts_logger = kwargs.pop("tts_logger")
        text_gen_logger = kwargs.pop("text_gen_logger")

        gpu = kwargs.get("device", "gpu") != "cpu"

        # Initialize STT
        self.asr = ASRProcessor(
            "openai/whisper-base.en",
            logger=asr_logger,
            **kwargs
        )
        asr_buffer_size = self.asr_buffer_length * 16_000
        self.asr_buffer = CircularBuffer(asr_buffer_size)

        # Initialize TTS
        self.tts = TTSProcessor(
            model_preset.tts_model_name,
            logger=tts_logger,
            gpu=gpu,
            speaker_name=model_preset.tts_speaker_name,
            **kwargs
        )

        # Initialize text gen
        self.text_gen = TextGenProcessor(
            model=model_preset.text_gen_model_name,
            logger=text_gen_logger,
            context=model_preset.text_gen_starting_context,
            **kwargs
        )

        self._run_thread = False
        self.queue: PriorityQueue[Tuple[
            datetime,
            str | bytes,
            Any,
        ]] = PriorityQueue()
        self.callback = callback
        self.current_input = ""
        self._cancel_current = False


    def start(self):
        self._run_thread = True
        thread = Thread(target=self._thread_loop)
        thread.daemon = True
        thread.start()

    def stop(self):
        self._run_thread = False

    def process_input(
        self,
        data: str | bytes,
        timestamp: datetime,
        callback_data: Any = None
    ) -> None:
        self.validate_queue(data)
        self.queue.put_nowait((timestamp, data, callback_data))

    def cancel_current_input(self) -> None:
        self._cancel_current = True

    def validate_queue(self, new_data: str | bytes = None) -> None:
        audio_bytes = len(new_data) if isinstance(new_data, bytes) else 0
        chars = len(new_data) if isinstance(new_data, str) else 0
        for data in list(self.queue.queue):
            if isinstance(data, str):
                chars += len(data)
            else:
                audio_bytes += len(data)

        if chars > self.max_queue_gen_chars:
            raise Full("No more generation chars can be added to the queue.")
        if audio_bytes > self.max_queue_audio_bytes:
            raise Full("No more audio can be added to the queue.")

    def _call_callback(self, *args):
        try:
            callback = self.callback
            callback(*args)
        except Exception: # pylint: disable=broad-exception-caught
            self.logger.error("Error during pipeline callback:\n%s", traceback.format_exc())

    def _thread_loop(self):
        while self._run_thread:
            try:
                timestamp, data, callback_data = self.queue.get(timeout=10)

                if self._cancel_current:
                    self._cancel_current = True
                    continue

                # Transcribe any audio and call respective callbacks
                if not isinstance(data, str):
                    self.current_input = ("bytes", str(len(data)))
                    self._call_callback("start", timestamp, self.current_input, callback_data)

                    # TODO: [asr] add timestamp to each transcription # eval if this is needed
                    transcription = self.asr.process_audio(data, self.asr_buffer)

                    if transcription != "":
                        data = transcription
                        self._call_callback("finish_asr", timestamp, transcription, callback_data)
                    else:
                        self._call_callback("finish", datetime.utcnow(), None, callback_data)
                        continue
                else:
                    self.current_input = ("text", data)
                    self._call_callback("start", timestamp, self.current_input, callback_data)

                self.current_input = ("text", data)

                # Generate AI response and TTS data
                # TODO: [generation] support streaming and kwargs
                ai_output = ""
                iterator = iter(self.text_gen.generate_from_prompt(data, stream=True))
                token = next(iterator, None)
                while token is not None:
                    ai_output += token
                    token = next(iterator, None)

                    # TODO: add support for eosen token
                    # If there is no next token (the sequence has ended), process the rest of
                    # the remaining tokens
                    if token is None:
                        sentence = ai_output
                    # else wait until a full sentence has been generated
                    elif (match := re.search(sentence_end_regex, ai_output)) is not None:
                        match_end = match.end()
                        sentence = ai_output[:match_end].rstrip()
                        ai_output = ai_output[match_end:]
                    else:
                        continue

                    # TODO: Consider sending just a message with whitespace rather than skip the
                    # event.
                    if sentence == "" or sentence.isspace():
                        continue

                    try:
                        tts_outputs = self.tts.tts(sentence)

                        if len(tts_outputs) == 1:
                            res = {
                                "wav": tts_outputs[0]["wav"],
                                "expressions": tts_outputs[0]["expressions"],
                                "text": sentence
                            }
                        else:
                            # In the unlikely chance that tts creates multiple outputs (due to the
                            # difference in how sentences are detected) then combine
                            expressions = []
                            for tts_output in tts_outputs:
                                expressions.extend(tts_output["expressions"])
                            wavs = [tts_output["wav"] for tts_output in tts_outputs]
                            res = {
                                "wav": np.concatenate(wavs),
                                "expressions": expressions,
                                "text": sentence
                            }

                        if self._cancel_current:
                            self._cancel_current = True
                            break

                        self._call_callback("finish_gen", datetime.utcnow(), res, callback_data)
                    except Exception: # pylint: disable=broad-exception-caught
                        self.logger.error(
                            "Error processing TTS with input `%s`:\n%s",
                            sentence,
                            traceback.format_exc()
                        )
                self._call_callback("finish", datetime.utcnow(), None, callback_data)
            except Empty:
                pass
            except Exception: # pylint: disable=broad-exception-caught
                self.logger.error("Error in pipeline thread:\n%s", traceback.format_exc())
