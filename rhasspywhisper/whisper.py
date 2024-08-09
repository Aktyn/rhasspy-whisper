import os
import wave
import uuid
from faster_whisper import WhisperModel


# Available models:
# tiny.en, tiny,
# base.en, base,
# small.en, small,
# medium.en, medium,
# large-v1, large-v2, large-v3, large,
# distil-large-v2, distil-medium.en, distil-small.en

class Whisper:
    def __init__(self, model_size='small.en'):
        self.model_size = model_size
        # self.model = WhisperModel(model_size, device="cuda", compute_type="float32", download_root='./models')
        # self.model = WhisperModel(model_size, device="cuda", compute_type="int8_float16", download_root='./models')
        # self.model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root='./models')
        self.model = WhisperModel(model_size, device="cpu", compute_type="auto", download_root='./models')

    def transcribe(self, audio_bytes: bytes, sample_rate: int):

        # TODO: test (maybe processing and saving to file with wave is not needed)
        # audio = np.frombuffer(audio_bytes, dtype=np.int16)
        # audio = audio.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

        file_path: str = f'./output/transcription_{uuid.uuid4()}.wav'
        wav_file: wave.Wave_write = wave.open(file_path, "wb")
        with wav_file:
            wav_file.setframerate(sample_rate)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframes(audio_bytes)

        # TODO: custom parameters
        segments, info = self.model.transcribe(
            file_path,
            language='en',
            task='transcribe',
            beam_size=5,
            suppress_blank=True,
            suppress_tokens=[],
            without_timestamps=True,
            word_timestamps=False,
            language_detection_segments=0,  # TODO: tweak for auto language detection
            # hotwords='' # TODO: add hotwords customizable by user
        )
        # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        # Remove processed file
        try:
            os.remove(file_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        output = ""
        for segment in segments:
            output += segment.text + "\n"

        return output
