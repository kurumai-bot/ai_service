import logging
import math
from typing import List, Union

import numpy as np
import numpy.typing as npt
import soundfile
import torch
from transformers import (
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)

from utils import CircularBuffer


class SileroVAD:
    def __init__(self) -> None:
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=True
        )

    def get_speech_timestamps(self):
        return self.utils[0]()

    def save_audio(self, path: str, tensor: torch.Tensor, sampling_rate: int = 16_000):
        return self.utils[1](path, tensor, sampling_rate)

    def read_audio(self, path: str, sampling_rate: int = 16_000):
        return self.utils[2](path, sampling_rate)

    def collect_chunks(self, tss: List[dict], wav: torch.Tensor):
        return self.utils[4](tss, wav)

    def get_chunk_confidence(
        self,
        audio: Union[npt.NDArray[np.float32], bytes],
        sampling_rate: int = 16_000,
    ) -> float:
        if isinstance(audio, bytes):
            return self.model(
                torch.frombuffer(audio, dtype=torch.float32), sampling_rate
            ).item()
        return self.model(torch.from_numpy(audio), sampling_rate).item()


class LiveASRInference:
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.device = kwargs.get("device", "cuda")

    def transcribe_buffer(self, buffer: np.ndarray) -> str:
        pass

    def transcribe_file(self, path: str) -> str:
        pass


class ASRProcessor:
    vad: SileroVAD = None

    def __init__(
        self,
        model: str | LiveASRInference,
        **kwargs
    ) -> None:
        self.logger = kwargs.pop("logger", None) or logging.getLogger("asr")

        self.vad_chunk_length: float = kwargs.pop("vad_chunk_length", 0.25)
        self.vad_confidence_threshhold: float = kwargs.pop("vad_confidence_threshhold", 0.4)

        # Initialize STT
        self.model: LiveASRInference = None
        if isinstance(model, str):
            if "wav2vec" in model:
                self.model = Wav2Vec(model, **kwargs)
            else:
                self.model = Whisper(model, **kwargs)
        else:
            self.model = model

        # Start VAD
        if type(self).vad is None:
            type(self).vad = SileroVAD()
        self.vad_chunk_size = math.ceil(self.vad_chunk_length * 16_000)

        self.logger.info(
            "ASR Processor initialized with model `%s`.",
            self.model.__class__.__name__
        )

    def process_audio(self, data: bytes, buffer: CircularBuffer = None) -> str:
        if buffer is None:
            buffer = CircularBuffer(5 * 16_000)

        # Fill the buffer with incoming messages
        float_array: np.ndarray = np.frombuffer(data, dtype=np.float32)

        transcriptions = []

        # Loop in vad chunk lengths, and check if they're speech
        i = 0
        last_chunk = None
        while i < len(float_array):
            # Also include if the next chunk will be smaller than the minimum chunk size
            if i + self.vad_chunk_size + (0.03 * 16_000) >= len(float_array):
                chunk = np.copy(float_array[i:])
                i = len(float_array)
            else:
                chunk = np.copy(float_array[i:i + self.vad_chunk_size])
                i += self.vad_chunk_size

            # Check if it is speech, if so add it to the buffer
            confidence = type(self).vad.get_chunk_confidence(chunk)
            if confidence >= self.vad_confidence_threshhold:
                # TODO: Implement batching for perf gains
                # Add last chunk since the vad tends to give really low confidences for leadings
                # sounds
                self.logger.debug("Transcribing buffer...")
                if last_chunk is not None:
                    for full_buffer in buffer.add(last_chunk):
                        transcriptions.append(self.model.transcribe_buffer(full_buffer))

                for full_buffer in buffer.add(chunk):
                    transcriptions.append(self.model.transcribe_buffer(full_buffer))
            # If not, then clear the buffer and put whatever was in there into the queue
            elif buffer.index > 0:
                self.logger.debug("Transcribing buffer...")
                transcriptions.append(self.model.transcribe_buffer(buffer.get()))
                buffer.clear()

            last_chunk = chunk

        # TODO: Test if there needs to be a diff join string here
        return "".join(transcriptions)


# NOTE: Whisper.cpp is not used here because the Python bindings linked in the readme at time of
# writing are 7/20/2023 significantly out of date, meaning there is no cuBLAS or quantize support.
# The last commits on the Cython bindings changed one line then undid the changes, and the last
# commits on the PyBind bindings were by bots, which ultimately does nothing since the submodule
# isn't used in the build process. So, in short, I'd have to write my own bindings which I simply do
# not have the time to maintain, nor do I want to expand the scope of this project even further.
class Whisper(LiveASRInference):
    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.speech_recognition_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            feature_extractor=self.processor.feature_extractor,
            tokenizer=self.processor.tokenizer,
            device=torch.device(self.device)
        )

    def transcribe_buffer(self, buffer: np.ndarray) -> str:
        transcription = self.speech_recognition_pipeline(buffer, max_new_tokens=10_000)

        return transcription["text"]

    def transcribe_file(self, path: str) -> str:
        audio_buffer, sample_rate = soundfile.read(path)
        assert sample_rate == 16_000
        assert audio_buffer.ndim == 1
        return self.transcribe_buffer(audio_buffer)


# TODO: combine asr processor into one class using pipelines
class Wav2Vec(LiveASRInference):
    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.to(self.device)

    def transcribe_buffer(self, buffer: np.ndarray) -> str:
        inputs = self.processor(
            torch.from_numpy(buffer),
            sampling_rate=16_000,
            return_tensors="pt",
            padding="longest",
        ).to(self.device).input_values

        with torch.no_grad():
            logits = self.model(inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription

    def transcribe_file(self, path: str) -> str:
        audio_buffer, _ = soundfile.read(path)
        return self.transcribe_buffer(audio_buffer)
