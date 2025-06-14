from pydub import AudioSegment

def merge_voice_with_bgm(source_audio_path, bgm_path, output_path):
    # Load the source audio and background music
    source_audio = AudioSegment.from_wav(source_audio_path)
    bgm = AudioSegment.from_wav(bgm_path)

    # Match background music duration if needed
    bgm = bgm[:len(source_audio)] if len(bgm) > len(source_audio) else bgm + AudioSegment.silent(duration=len(source_audio)-len(bgm))

    # Mix with proper gain to avoid overpowering
    mixed = source_audio.overlay(bgm - 6)
    mixed.export(output_path, format="wav")
