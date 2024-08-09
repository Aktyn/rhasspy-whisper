"""Command-line interface to rhasspywhisper."""
import io
import logging
import sys
import typing
import wave
from pathlib import Path

from . import WebRtcVadRecorder
from .const import VoiceCommandEventType, OutputType
from .utils import trim_silence
from .args import parse_args
from .whisper import Whisper

_LOGGER = logging.getLogger("rhasspywhisper")


def main():
    """Main entry point."""
    args = parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    if args.quiet or (args.trim_silence and not args.split_dir and args.output_type != OutputType.TRANSCRIPTION):
        args.output_type = OutputType.NONE

    if args.split_dir:
        # Directory to write WAV file(s) split by silence
        args.split_dir = Path(args.split_dir)
        args.split_dir.mkdir(parents=True, exist_ok=True)

    whisper = None
    if args.output_type == OutputType.TRANSCRIPTION:
        whisper = Whisper()

    print("Reading raw 16Khz mono audio from stdin...", file=sys.stderr)

    try:
        recorder = WebRtcVadRecorder(
            max_seconds=args.max_seconds,
            vad_mode=args.sensitivity,
            skip_seconds=args.skip_seconds,
            min_seconds=args.min_seconds,
            speech_seconds=args.speech_seconds,
            silence_seconds=args.silence_seconds,
            before_seconds=args.before_seconds,
            silence_method=args.silence_method,
            current_energy_threshold=args.current_threshold,
            max_energy=args.max_energy,
            max_current_ratio_threshold=args.max_current_ratio_threshold,
        )

        dynamic_max_energy = args.max_energy is None
        max_energy: typing.Optional[float] = args.max_energy
        split_index = 0

        recorder.start()

        while True:
            chunk = sys.stdin.buffer.read(args.chunk_size)
            if not chunk:
                break

            result = recorder.process_chunk(chunk)
            output = ""

            if args.output_type != OutputType.NONE and args.output_type != OutputType.TRANSCRIPTION:
                # Print voice command events
                for event in recorder.events:
                    if event.type == VoiceCommandEventType.STARTED:
                        output += "["
                    elif event.type == VoiceCommandEventType.STOPPED:
                        output += "]"
                    elif event.type == VoiceCommandEventType.SPEECH:
                        output += "S"
                    elif event.type == VoiceCommandEventType.SILENCE:
                        output += "-"
                    elif event.type == VoiceCommandEventType.TIMEOUT:
                        output += "T"

                recorder.events.clear()

                # Print speech/silence
                if args.output_type == OutputType.SPEECH_SILENCE:
                    if recorder.last_speech:
                        output += "!"
                    else:
                        output += "."
                elif args.output_type == OutputType.CURRENT_ENERGY:
                    # Debiased energy of current chunk
                    energy = int(WebRtcVadRecorder.get_debiased_energy(chunk))
                    output += f"{energy} "
                elif args.output_type == OutputType.MAX_CURRENT_RATIO:
                    # Ratio of max/current energy
                    energy = WebRtcVadRecorder.get_debiased_energy(chunk)
                    if dynamic_max_energy:
                        if max_energy is None:
                            max_energy = energy
                        else:
                            max_energy = max(energy, max_energy)

                    assert max_energy is not None, "Max energy not set"
                    ratio = max_energy / energy if energy > 0.0 else 0.0
                    output += f"{ratio:.2f} "

                print(output, end="", flush=True)

            if result:
                # Reset after voice command
                audio_bytes = recorder.stop()

                if args.output_type == OutputType.TRANSCRIPTION:
                    # Split audio
                    if args.trim_silence:
                        audio_bytes = trim_silence(
                            audio_bytes,
                            chunk_size=args.trim_chunk_size,
                            ratio_threshold=args.trim_ratio,
                            keep_chunks_before=args.trim_keep_before,
                            keep_chunks_after=args.trim_keep_after,
                        )

                    try:
                        transcription = whisper.transcribe(audio_bytes, recorder.sample_rate)
                        _LOGGER.info(transcription)
                    except Exception as e:
                        _LOGGER.error(e)
                    # split_index += 1
                elif args.split_dir:
                    # Split audio
                    if args.trim_silence:
                        audio_bytes = trim_silence(
                            audio_bytes,
                            chunk_size=args.trim_chunk_size,
                            ratio_threshold=args.trim_ratio,
                            keep_chunks_before=args.trim_keep_before,
                            keep_chunks_after=args.trim_keep_after,
                        )

                    split_wav_path = args.split_dir / args.split_format.format(
                        split_index
                    )

                    wav_file: wave.Wave_write = wave.open(str(split_wav_path), "wb")
                    with wav_file:
                        wav_file.setframerate(recorder.sample_rate)
                        wav_file.setsampwidth(2)
                        wav_file.setnchannels(1)
                        wav_file.writeframes(audio_bytes)

                    _LOGGER.info("Wrote %s", split_wav_path)
                    split_index += 1
                elif args.trim_silence:
                    # Trim silence without splitting
                    audio_bytes = trim_silence(
                        audio_bytes,
                        chunk_size=args.trim_chunk_size,
                        ratio_threshold=args.trim_ratio,
                        keep_chunks_before=args.trim_keep_before,
                        keep_chunks_after=args.trim_keep_after,
                    )

                    with io.BytesIO() as wav_io:
                        wav_file: wave.Wave_write = wave.open(wav_io, "wb")
                        with wav_file:
                            wav_file.setframerate(recorder.sample_rate)
                            wav_file.setsampwidth(2)
                            wav_file.setnchannels(1)
                            wav_file.writeframes(audio_bytes)

                        sys.stdout.buffer.write(wav_io.getvalue())

                    break

                recorder.start()

    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
