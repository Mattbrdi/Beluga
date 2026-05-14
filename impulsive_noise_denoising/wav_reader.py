import struct
import os

import numpy as np 

def read_wav_metadata_and_data(file_path):
    """Read basic WAV metadata plus raw audio bytes for PCM or float WAV files."""
    with open(file_path, 'rb') as wav_file:
        riff = wav_file.read(12)
        if len(riff) != 12 or riff[:4] != b'RIFF' or riff[8:12] != b'WAVE':
            raise ValueError("Expected a RIFF/WAVE file")

        fmt_chunk = None
        audio_data = None
        while True:
            chunk_header = wav_file.read(8)
            if len(chunk_header) == 0:
                break
            if len(chunk_header) != 8:
                raise ValueError("Invalid WAV chunk header")

            chunk_id, chunk_size = struct.unpack('<4sI', chunk_header)
            chunk_data = wav_file.read(chunk_size)
            if chunk_size % 2 == 1:
                wav_file.seek(1, os.SEEK_CUR)

            if chunk_id == b'fmt ':
                fmt_chunk = chunk_data
            elif chunk_id == b'data':
                audio_data = chunk_data

            if fmt_chunk is not None and audio_data is not None:
                break

    if fmt_chunk is None or audio_data is None:
        raise ValueError("Could not find both fmt and data chunks in WAV file")
    if len(fmt_chunk) < 16:
        raise ValueError("Invalid WAV fmt chunk")

    audio_format, n_channels, frame_rate, _, block_align, bits_per_sample = struct.unpack('<HHIIHH', fmt_chunk[:16])
    if audio_format == 65534 and len(fmt_chunk) >= 40:
        subformat = fmt_chunk[24:40]
        if subformat.startswith(b'\x01\x00'):
            audio_format = 1
        elif subformat.startswith(b'\x03\x00'):
            audio_format = 3

    sample_width = bits_per_sample // 8
    if block_align != n_channels * sample_width:
        raise ValueError(f"Unsupported WAV block alignment: {block_align}")

    return audio_data, audio_format, n_channels, sample_width, frame_rate

def read_wav_file(file_path, num_canals = 4):
    """Read a PCM WAV file of 4 sources and convert it to a centered NumPy array.

    Args:
        file_path (Path) : Audio file path

    Returns:
        tuple(np.ndarray, int): Audio data, frame rate
    """
    audio_data, audio_format, n_channels, sample_width, frame_rate = read_wav_metadata_and_data(file_path)
    if sample_width not in (2, 3, 4):
        raise ValueError(f"Audio width should be 16, 24, or 32 bits, but got {sample_width * 8} bits")
    if audio_format not in (1, 3):
        raise ValueError(f"Expected PCM or IEEE float WAV format, but got format tag {audio_format}")
    if audio_format == 3 and sample_width != 4:
        raise ValueError(f"Expected 32-bit float WAV data, but got {sample_width * 8} bits")
    if n_channels != num_canals:
        raise ValueError(f"Expected a tetrahedral microphone recording with {num_canals} channels, but got {n_channels} channels")

    # NumPy array conversion
    if audio_format == 3:
        tetra_array = np.frombuffer(audio_data, dtype='<f4').astype(np.float64)
    elif sample_width == 2:
        tetra_array = np.frombuffer(audio_data, dtype='<i2').astype(np.float64)
        tetra_array /= 2 ** 15
    elif sample_width == 3:
        raw = np.frombuffer(audio_data, dtype=np.uint8).reshape(-1, 3)
        tetra_array = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        tetra_array = np.where(tetra_array >= 2**23, tetra_array - 2**24, tetra_array).astype(np.float64)
        tetra_array /= 2 ** 23
    else:
        tetra_array = np.frombuffer(audio_data, dtype='<i4').astype(np.float64)
        tetra_array /= 2 ** 31

    # Reshape to avoid interleaving channels
    tetra_array = tetra_array.reshape(-1, num_canals).T
    return tetra_array, frame_rate
    
def extract_time_slice(tetra_array, frame_rate, start_time, end_time):
    n_channels, length = np.shape(tetra_array)
    duration = length / frame_rate
    if not start_time or not end_time:
        return tetra_array
    if start_time < 0:
        raise ValueError(f"Provided start time should be at least zero, but got {start_time}")
    if start_time >= duration:
        raise ValueError(f"Provided start time should be less than duration, but got start_time: {start_time}s and duration: {duration}s")
    
    if end_time <= 0:
        raise ValueError(f"Provided end time should be greater than zero, but got {end_time}")
    if end_time <= start_time:
        raise ValueError(f"Provided end time should be greater than start time, but got start_time: {start_time}s and end_time: {end_time}s")
    if end_time > duration:
        raise ValueError(f"Provided end time should be less than or equal to duration, but got end_time: {end_time}s and duration: {duration}s")
    start_idx = int(round(start_time * frame_rate))
    end_idx = int(round(end_time * frame_rate))
    return tetra_array[:, start_idx:end_idx]
