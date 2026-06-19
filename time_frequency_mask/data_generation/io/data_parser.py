import numpy as np 
from numpy.typing import NDArray

import os
import wave
from struct import unpack
from pathlib import Path
from PIL import Image

from time_frequency_mask.configuration import WHISTLE_MASK_PATH, WHISTLE_WAV_PATH, SAMPLING_RATE, MIN_FREQ, MAX_FREQ, N_FFT, HOP_LENGTH
from time_frequency_mask.stft import scipy_spectrogram, scipy_db_spectrogram, frequency_band

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

            chunk_id, chunk_size = unpack('<4sI', chunk_header)
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

    audio_format, n_channels, frame_rate, _, block_align, bits_per_sample = unpack('<HHIIHH', fmt_chunk[:16])
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

    if num_canals == 1:
        tetra_array = tetra_array[0]
    return tetra_array, frame_rate

def read_image_file(file_path : str) -> NDArray:
    data = Image.open(file_path)
    width, height  = data.size
    data = np.array(data.get_flattened_data(), dtype=np.uint8)
    data = data.reshape((height, width))
    return data 

def retrieve_wav_and_masks_paths() -> dict[str, list[Path]]:
    wav_dir = Path(WHISTLE_WAV_PATH)
    wav_paths = sorted(wav_dir.glob("*.wav"), key=lambda x: int(x.stem[7:]))

    mask_dir = Path(WHISTLE_MASK_PATH)
    mask_paths = sorted(mask_dir.glob("*.png"), key=lambda x: int(x.stem    [7:-5]))

    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found in {wav_dir}")

    if not mask_paths:
        raise FileNotFoundError(f"No WAV files found in {mask_dir}")

    wav_dir_stems = np.array([wav_path.stem for wav_path in wav_paths])
    mask_dir_stems = np.array([mask_path.stem for mask_path in mask_paths])

    mask_dir_stems_trimmed = np.array([stem[:-5] for stem in mask_dir_stems])

    available_whistles = np.array(wav_dir_stems)[
        np.isin(wav_dir_stems, mask_dir_stems_trimmed, assume_unique=True)
    ]
    
    available_whistles_masks = np.array(mask_dir_stems)[
        np.isin(wav_dir_stems, mask_dir_stems_trimmed, assume_unique=True)
    ]

    wav_paths = [wav_dir / f"{available_whistle}.wav" for available_whistle in available_whistles]
    mask_paths = [mask_dir / f"{available_whistle_mask}.png" for available_whistle_mask in available_whistles_masks]

    return {"wav_paths": wav_paths, "mask_paths": mask_paths}

def save_wav_file(audio_array : list[NDArray[np.float64]], file_name : str, num_channels = 4, sampling_rate : float = SAMPLING_RATE):
    """Save array to wav file

    Parameters
    ----------
    array : NDArray[np.float64]
        array saved as wav
    sampling_rate : float
        
    format : str
        format to save the wav on, choose from the following :
    """
    
    audio = np.asarray(audio_array, dtype=np.float64)

    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    
    if audio.shape[0]!= num_channels:
        raise ValueError(f"Incorrect tetrahedra num of channels got {len(audio_array)} instead of {num_channels}")
    
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio /= peak

    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    interleaved = pcm.T

    with wave.open(file_name, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sampling_rate))
        wav_file.writeframes(interleaved.tobytes()) 

def save_mask(mask : NDArray[np.uint8], output_path : str):
    mask = np.asarray(mask).astype(bool)

    mask = mask.astype(np.uint8) * 255

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(mask).save(str(output_path))

def save_stft_png(audio_array: list[NDArray[np.float64]], output_path : str, sampling_rate = SAMPLING_RATE, is_db=False, fmin=MIN_FREQ, fmax=MAX_FREQ, n_fft = N_FFT, hop_length = HOP_LENGTH):
    Ds = []
    freqs_list = []
    times_list = []

    import matplotlib.pyplot as plt
    vmin = -120
    vmax = -20
    fig, axs = plt.subplots(4, 1, figsize=(30, 20), sharex=True, sharey=True)
    for canal in audio_array:
        if is_db:
            freqs, times, D = scipy_db_spectrogram(canal, sampling_rate, n_fft, hop_length)
        else:
            freqs, times, D = scipy_spectrogram(canal, sampling_rate, n_fft, hop_length)
      
        freqs, D = frequency_band(freqs, D, fmin, fmax)
    
        D = D - np.min(D)
        if not np.all(D == 0):
            D = D / np.percentile(np.abs(D), 99)
        D = np.array(plt.cm.magma(D))
        D = np.round(D*255)
        D = D.astype(np.uint8)
        Ds.append(D)
        freqs_list.append(freqs)
        times_list.append(times)


    for i, D in enumerate(Ds):
        freqs = freqs_list[i]
        times = times_list[i]
        
        extent = [
            times[0],
            times[-1] if len(times) > 1 else 0,
            freqs[0],
            freqs[-1],
        ]

        axs[i].imshow(
            D,
            origin='lower',
            aspect='auto',
            extent=extent,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )

        axs[i].set_title(f'Spectrogramme SciPy du Canal {i+1}')
        axs[i].set_ylim([fmin, fmax])
        axs[i].set_ylabel('Frequency (Hz)')
    plt.savefig(output_path)
    plt.close(fig)
