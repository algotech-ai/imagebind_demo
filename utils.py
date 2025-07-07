import ffmpeg

def copy_audio_wav(video_path: str, output_wav: str):
    """
    Extract audio from video file to 16-bit PCM WAV format.

    Args:
        video_path: Path to input video

    Returns:
        BinaryIO: Temporary file containing WAV audio
    """

    (
        ffmpeg.input(video_path)
        .output(
            output_wav,
            acodec="pcm_s16le",  # 16-bit PCM
            ac=1,  # mono
            ar=16000,  # 16kHz sample rate
            vn=None,  # no video
        )
        .overwrite_output()
        .run(quiet=True)
    )
