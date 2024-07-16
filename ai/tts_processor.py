import itertools
import logging
import math
from typing import Any, Dict, Iterable, List, Tuple

from coqpit import Coqpit
import numpy as np
from TTS.api import TTS
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.synthesis import  synthesis, trim_silence
from TTS.tts.utils.text.characters import BaseCharacters
from TTS.tts.utils.text.phonemizers import DEF_LANG_TO_PHONEMIZER, get_phonemizer_by_name
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio.numpy_transforms import save_wav


# I seriously don't know where these come from and I couldn't find a good source
# listing all of them in the espeak docs of espeak-ng source, so some may be
# redundant.
UTILITY_PHONEMES = {
    "ˈ",
    "ˌ",
    "%",
    "=",
    "_:",
    "_",
    "|",
    "ː",
    " ",
    "\\",
    "!",
    ".",
    ",",
    "?",
    "\"",
    ":",
    "̩ "
}

VISEME_INDEX_TO_NAME = {
    0:  "sil",
    1:  "PP",
    2:  "FF",
    3:  "TH",
    4:  "DD",
    5:  "kk",
    6:  "CH",
    7:  "SS",
    8:  "nn",
    9:  "RR",
    10: "aa",
    11: "E",
    12: "I",
    13: "O",
    14: "U",
}
VISEME_NAME_TO_INDEX = {value: key for key, value in VISEME_INDEX_TO_NAME.items()}


class Viseme:
    # Let viseme argument be index or name and figure out based off that
    def __init__(self, viseme: str | int, weight: float = 1) -> None:
        if isinstance(viseme, str):
            try:
                self.index = int(viseme)
                self.name = VISEME_INDEX_TO_NAME[self.index]
            except ValueError:
                self.name = viseme
                self.index = VISEME_NAME_TO_INDEX[self.name]
        else:
            self.index = viseme
            self.name = VISEME_INDEX_TO_NAME[self.index]

        self.weight = weight

    def __repr__(self) -> str:
        return f"{self.name}:{self.weight:.2f}"


# OVR Lip Sync visemes
# https://developer.oculus.com/documentation/unity/audio-ovrlipsync-viseme-reference/
# The docs don't contain all possible phonemes a viseme can associate with, so
# I cross referenced the OVR Lip Sync demo and these Microsoft docs:
# https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/speech-ssml-phonetic-sets
# https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/how-to-speech-synthesis-viseme
# which contain more expansive information on phoneme to viseme conversions,
# albeit with Microsoft's different viseme set.
# Microsoft's docs didn't list all possible phonemes that espeak will spit out,
# so I also added some phonemes and their about equivalent viseme.
PHONEME_TO_VISEME = {
    "p":  Viseme(1),
    "b":  Viseme(1),
    "m":  Viseme(1),
    "f":  Viseme(2),
    "v":  Viseme(2),
    "θ":  Viseme(3),
    "ð":  Viseme(3),
    "n":  Viseme(4),
    "t":  Viseme(4),
    "d":  Viseme(4),
    "ɾ":  Viseme(4),
    "ʔ":  Viseme(4),
    "k":  Viseme(5),
    "g":  Viseme(5),
    "ɡ":  Viseme(5),
    "ŋ":  Viseme(5),
    "tʃ": Viseme(6),
    "dʒ": Viseme(6),
    "ʃ":  Viseme(6),
    "ʒ":  Viseme(6),
    "s":  Viseme(7),
    "z":  Viseme(7),
    "l":  Viseme(8),
    "ɫ":  Viseme(8),
    "ɹ":  Viseme(9),
    "ɝ":  Viseme(9),
    "ɚ":  Viseme(9),
    "ɑ":  Viseme(10),
    "ə":  Viseme(10),
    "ʌ":  Viseme(10),
    "ɐ":  Viseme(10),
    "ɛ":  Viseme(11),
    "æ":  Viseme(11),
    "j":  Viseme(12),
    "ɪ":  Viseme(12),
    "i":  Viseme(12),
    "ᵻ":  Viseme(12),
    "o":  Viseme(13),
    "ɔ":  Viseme(13),
    "w":  Viseme(14),
    "u":  Viseme(14),
    "ʊ":  Viseme(14),
    "h":  [Viseme(1, weight=0.5), Viseme(5, weight=0.5)],
    "ɜ":  [Viseme(9, weight=0.5), Viseme(10, weight=0.5)],
    "aɪ": [Viseme(10, weight=0.5), Viseme(11, weight=0.5)],
    "aʊ": [Viseme(10, weight=0.5), Viseme(11, weight=0.5)],
    "eɪ": [Viseme(11, weight=0.5), Viseme(12, weight=0.5)],
    "ɔɪ": [Viseme(12, weight=0.25), Viseme(13, weight=0.75)],
}


class TTSProcessor:
    def __init__(self, model: str | TTS, **kwargs) -> None:
        self.logger = kwargs.pop("logger", None) or logging.getLogger("tts")

        self.speaker_name: str = kwargs.get("speaker_name", None)
        self.language: str = kwargs.get("language", None)
        self.gpu: bool = kwargs.get("gpu", True)

        # Start TTS
        if isinstance(model, str):
            self.tts_wrapper = TTS(model, progress_bar=False)
        else:
            self.tts_wrapper = model

        if self.gpu:
            self.tts_wrapper.to("cuda")

        self.synthesizer = self.tts_wrapper.synthesizer
        self.sample_rate: int = self.synthesizer.output_sample_rate
        self.tts_config: Coqpit = self.synthesizer.tts_config
        self.tts_model: BaseTTS = self.synthesizer.tts_model
        self.tokenizer: TTSTokenizer = self.tts_model.tokenizer
        self.characters: BaseCharacters = self.tokenizer.characters

        # This simplifies speech synthesis greatly
        if self.synthesizer.vocoder_model:
            self.logger.warning((
                "Model has vocoder specified which is unsupported. Griffin LIM will be used "
                "instead."
            ))

        # Start phonemizer
        if not self.tokenizer.phonemizer:
            self.phonemizer = get_phonemizer_by_name(
                DEF_LANG_TO_PHONEMIZER[self.language],
                language=self.language
            )
            self.logger.warning("Model doesn't use phonemizer. Visemes will not be supported.")
        else:
            self.phonemizer = self.tokenizer.phonemizer

        self.logger.info("TTS Processor initialized with model `%s`.", self.tts_wrapper.model_name)

    def tts_to_wav(self, text: str, path: str, sentence_pause: float = 0.5) -> None:
        wav = self.tts_to_buffer(text, sentence_pause=sentence_pause)
        save_wav(wav=wav, path=path, sample_rate=self.sample_rate)

    def save_wav(self, wav: np.ndarray, path: str):
        save_wav(wav=wav, path=path, sample_rate=self.sample_rate)

    def tts_to_buffer(self, text: str, sentence_pause: float = 0.5) -> np.ndarray:
        outputs = self.tts(text, sentence_pause=sentence_pause)

        if len(outputs) == 1:
            return outputs[0]["wav"]

        # Find number of bytes
        length = 0
        for output in outputs:
            length += len(output["wav"])

        # Create buffer based off number of bytes
        buffer = np.empty(length, dtype=outputs[0]["wav"].dtype)
        i = 0

        # Actually add data to buffer
        for output in outputs:
            buffer[i:i + len(output["wav"])] = output["wav"]
            i += len(output["wav"])

        return buffer

    def tts_to_bytes(self, text: str, sentence_pause: float = 0.5) -> bytes:
        return self.tts_to_buffer(text, sentence_pause=sentence_pause).tobytes()

    def tts(self, text: str, sentence_pause: float = 0.5) -> List[Dict[str, Any]]:
        # Get internal value for speaker
        if self.tts_wrapper.is_multi_speaker and self.speaker_name:
            speaker_id = self.synthesizer.tts_model.speaker_manager.name_to_id[self.speaker_name]
        elif self.synthesizer.tts_model.speaker_manager:
            speaker_id = next(iter(self.synthesizer.tts_model.speaker_manager.name_to_id.values()))
        else:
            speaker_id = None

        # Get internal value for language
        if self.tts_wrapper.is_multi_lingual and self.language:
            language_id = self.tts_model.language_manager.name_to_id[self.language]
        elif self.tts_model.language_manager:
            language_id = next(iter(self.tts_model.language_manager.name_to_id.values()))
        else:
            language_id = None

        # Check whether to trim silence
        do_trim_silence = "do_trim_silence" in self.tts_config.audio \
            and self.tts_config.audio["do_trim_silence"]
        outputs = []
        for sentence in self.synthesizer.split_into_sentences(text):
            # Using this rather than the wrapper because the wrapper strips out
            # a lot of important information
            # Run text through the model
            output = synthesis(
                self.synthesizer.tts_model,
                sentence,
                CONFIG=self.synthesizer.tts_config,
                use_cuda=self.gpu,
                speaker_id=speaker_id,
                use_griffin_lim=True,
                do_trim_silence=False,
                language_id=language_id
            )

            output["text"] = sentence
            output["wav"] = output["wav"].squeeze()

            expressions = None
            if self.tokenizer.use_phonemes:
                # Based off implimentation in this repo:
                # https://github.com/ManimCommunity/manim-voiceover/blob/0726b6909b98ec57c82318d85cacc5da749b6008/manim_voiceover/services/coqui/utils_synthesizer.py#L459
                # Calculate what text is pronounced at what time
                alignments = output["alignments"]
                if alignments.dim() == 3:
                    alignments = alignments[0]
                max_indices = alignments.argmax(dim=1).tolist()

                input_tokens = output["text_inputs"].to("cpu").numpy().squeeze()

                expressions = self.get_expressions_from_tokens(input_tokens, max_indices)
            output["expressions"] = expressions

            if do_trim_silence:
                trim_silence(output["wav"], self.tts_model.ap)

            # Add some space between sentences
            output["wav"] = np.pad(
                output["wav"],
                (0, math.ceil(self.sample_rate * sentence_pause)),
                mode="constant",
                constant_values=0
            )
            outputs.append(output)

        return outputs

    def get_phonemes(self, text: str, separator: str = "|") -> str:
        return self.phonemizer.phonemize(text, separator, self.language)

    # def get_expressions_from_phonemes(
    #     self,
    #     phonemes: str,
    #     separator: str = "|"
    # ) -> List[Tuple[float, Viseme | List[Viseme]]]:
    #     # Loop through phoneme string and get largest phoneme it could match
    #     current_phoneme = ""
    #     expressions = []
    #     for phoneme in phonemes:
    #         # Ignore utility phonemes for now
    #         if current_phoneme in UTILITY_PHONEMES:
    #             continue

    #         last_phoneme = current_phoneme
    #         current_phoneme += phoneme
    #         if last_phoneme in PHONEME_TO_VISEME and current_phoneme not in PHONEME_TO_VISEME:
    #             # If last phoneme is a viseme, and the current one isn't,
    #             # set current_phoneme to char and add last_phoneme's viseme
    #             expressions.append((-1, PHONEME_TO_VISEME[last_phoneme]))
    #             current_phoneme = phoneme

    #         clean_phoneme = self._check_phoneme_for_separator(current_phoneme, separator)
    #         if len(clean_phoneme) > 0:
    #             if clean_phoneme not in PHONEME_TO_VISEME:
    #                 # If current_phoneme hasn't been cleared and isn't a
    #                 # valid phoneme, then it isn't recognized
    #                 self.logger.warning("Unrecognized phoneme `%s`", clean_phoneme)
    #             else:
    #                 # Else just add the viseme
    #                 expressions.append((-1, PHONEME_TO_VISEME[clean_phoneme]))

    #             # Clear phoneme
    #             current_phoneme = ""

    #     return expressions

    def get_expressions_from_tokens(
        self,
        tokens: List[int],
        alignment: Iterable = None
    ) -> List[Tuple[float, Viseme | List[Viseme]]]:
        if len(alignment) != len(tokens):
            raise ValueError("`tokens` and `alignment` must be the same length.")

        # Create dummy iterator so loop doesn't error
        if alignment is None:
            alignment = itertools.repeat(-1, len(tokens))
            hop_length = 1
        else:
            hop_length = self.tts_config.audio["hop_length"]

        # Loop through phoneme string and get largest phoneme it could match
        current_phoneme = ""
        last_hop_offset = 0
        expressions = []
        for token, hop_offset in zip(tokens, alignment):
            characters = self.characters
            phoneme = characters.id_to_char(token)

            # Ignore utility phonemes and special tokens for now
            if (phoneme in UTILITY_PHONEMES
                    or token == characters.pad_id
                    or token == characters.blank_id
                    or (characters.eos and token == characters.char_to_id(characters.eos))
                    or (characters.bos and token == characters.char_to_id(characters.bos))):
                continue

            # Treat spaces as separators
            if len(current_phoneme) > 0 and phoneme.isspace():
                if current_phoneme not in PHONEME_TO_VISEME:
                    # If current_phoneme hasn't been cleared and isn't a
                    # valid phoneme, then it's unrecognized
                    self.logger.warning("Unrecognized phoneme `%s`", current_phoneme)
                else:
                    # Else just add the viseme
                    expressions.append((
                        (last_hop_offset * hop_length) / self.sample_rate,
                        PHONEME_TO_VISEME[last_phoneme]
                    ))

                # Clear phoneme
                current_phoneme = ""

            last_phoneme = current_phoneme
            current_phoneme += phoneme
            if last_phoneme in PHONEME_TO_VISEME and current_phoneme not in PHONEME_TO_VISEME:
                # If last phoneme is a viseme, and the current one isn't,
                # set current_phoneme to char and add last_phoneme's viseme
                expressions.append((
                    (last_hop_offset * hop_length) / self.sample_rate,
                    PHONEME_TO_VISEME[last_phoneme]
                ))
                current_phoneme = phoneme
            last_hop_offset = hop_offset

        # Check if whatever remains is also a phoneme
        if current_phoneme in PHONEME_TO_VISEME:
            expressions.append((
                (last_hop_offset * hop_length) / self.sample_rate,
                PHONEME_TO_VISEME[current_phoneme]
            ))

        return expressions

    def _check_phoneme_for_separator(self, phoneme: str, separator: str) -> str:
        characters = self.characters
        # Yea all these substrings do copying but the speed really shouldn't
        # matter
        if phoneme[-1].isspace():
            return phoneme[:-1]
        if separator and phoneme[-len(separator)] == separator:
            return phoneme[-len(separator)]
        if characters.blank and phoneme[-len(characters.blank)] == characters.blank:
            return phoneme[-len(characters.blank)]
        if characters.pad and phoneme[-len(characters.pad)] == characters.pad:
            return phoneme[-len(characters.pad)]
        if characters.bos and phoneme[-len(characters.bos)] == characters.bos:
            return phoneme[-len(characters.bos)]
        if phoneme[-len(characters.eos)] == characters.eos:
            return phoneme[-len(characters.eos)]
        return phoneme


if __name__ == "__main__":
    TXT = "the quick brown fox jumps over the lazy dog"
    tts = TTSProcessor("tts_models/en/vctk/vits", speaker_name="p300", gpu=True)
    tts_output = tts.tts(TXT)
    tts.save_wav(tts_output[0]["wav"], "output.wav")
    print(tts.get_phonemes(TXT))
    print([tts.characters.id_to_char(token) for token in tts_output[0]["text_inputs"].to("cpu").numpy().squeeze()])
    print(tts_output[0]["expressions"])
    print(tts.tts_config.audio["hop_length"])
    print(tts_output[0]["alignments"][0].argmax(dim=1).tolist())
