import argparse

from .const import SilenceMethod, OutputType


def parse_args():
    parser = argparse.ArgumentParser(prog="rhasspy-whisper")
    parser.add_argument(
        "--output-type",
        default=OutputType.TRANSCRIPTION,
        choices=[e.value for e in OutputType],
        help="Type of printed output",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=960,
        help="Size of audio chunks. Must be 10, 20, or 30 ms for VAD.",
    )
    parser.add_argument(
        "--skip-seconds",
        type=float,
        default=0.0,
        help="Seconds of audio to skip before a voice command",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="Maximum number of seconds for a voice command",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=1.0,
        help="Minimum number of seconds for a voice command",
    )
    parser.add_argument(
        "--speech-seconds",
        type=float,
        default=0.3,
        help="Consecutive seconds of speech before start",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=0.5,
        help="Consecutive seconds of silence before stop",
    )
    parser.add_argument(
        "--before-seconds",
        type=float,
        default=0.5,
        help="Seconds to record before start",
    )
    parser.add_argument(
        "--sensitivity",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="VAD sensitivity (1-3)",
    )
    parser.add_argument(
        "--current-threshold",
        type=float,
        help="Debiased energy threshold of current audio frame",
    )
    parser.add_argument(
        "--max-energy",
        type=float,
        help="Fixed maximum energy for ratio calculation (default: observed)",
    )
    parser.add_argument(
        "--max-current-ratio-threshold",
        type=float,
        help="Threshold of ratio between max energy and current audio frame",
    )
    parser.add_argument(
        "--silence-method",
        choices=[e.value for e in SilenceMethod],
        default=SilenceMethod.VAD_ONLY,
        help="Method for detecting silence",
    )

    # Splitting and trimming by silence
    parser.add_argument(
        "--split-dir",
        help="Split incoming audio by silence and write WAV file(s) to directory",
    )
    parser.add_argument("--audio-output-dir",
                        default="output",
                        help="Similar to --split-dir, but for temporary audio files used for whisper")
    parser.add_argument(
        "--split-format",
        default="{}.wav",
        help="Format for split file names (default: '{}.wav', only with --split-dir)",
    )
    parser.add_argument(
        "--trim-silence",
        action="store_true",
        help="Trim silence when splitting (only with --split-dir)",
    )
    parser.add_argument(
        "--trim-ratio",
        default=20.0,
        type=float,
        help="Max/current energy ratio used to detect silence (only with --trim-silence)",
    )
    parser.add_argument(
        "--trim-chunk-size",
        default=960,
        type=int,
        help="Size of audio chunks for detecting silence (only with --trim-silence)",
    )
    parser.add_argument(
        "--trim-keep-before",
        default=0,
        type=int,
        help="Number of audio chunks before speech to keep (only with --trim-silence)",
    )
    parser.add_argument(
        "--trim-keep-after",
        default=0,
        type=int,
        help="Number of audio chunks after speech to keep (only with --trim-silence)",
    )

    parser.add_argument("--quiet", action="store_true", help="Set output type to none")

    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )

    parser.add_argument("--models-dir", default="models",
                        help="Directory for automatically downloading whisper models")

    return parser.parse_args()
