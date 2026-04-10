"""Module to import/export wav files into python."""
import wave
import numpy as np
import os
import datetime
from src.utils.sub_classes import AudioArray, AudioMetadata, Tetrahedra

########## Basic wav functions ###########

def read_wav_file(file_path):
    """Read a 16 bit audio file of 4 sources with WAV format and convert it to a centered NumPy array

    Args:
        file_path (Path) : Audio file path

    Returns:
        tuple(np.ndarray, int): Audio data, frame rate
    """
    with wave.open(str(file_path), 'rb') as wav_file:
        # Read file information
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        if sample_width != 2:
            raise ValueError("Audio width should be 16 bits, but got {sample_width} of sample width")
        if n_channels != 4:
            raise ValueError(f"Expected a tetrahedral microphone recording with 4 channels, but got {n_channels} channels")
        audio_data = wav_file.readframes(n_frames)

        # NumPy array conversion
        tetra_array = np.frombuffer(audio_data, dtype=np.int16)
        # Reshape to avoid interleaving channels
        tetra_array = (tetra_array.reshape(-1, n_channels).T).astype(float)
        tetra_array /= 32768. # Values between [-1,1]
        return tetra_array, frame_rate
    
def read_mono_file(file_path):
    """Read a 16 bit audio file of 1 source with WAV format and convert it to a centered NumPy array

    Args:
        file_path (Path) : Audio file path

    Returns:
        tuple(np.ndarray, int): Audio data, frame rate
    """
    with wave.open(str(file_path), 'rb') as wav_file:
        # Read file information
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        if sample_width != 2:
            raise ValueError("Audio width should be 16 bits, but got {sample_width} of sample width")
        if n_channels != 1:
            raise ValueError(f"Expected a sing microphone channel, but got {n_channels} channels")
        audio_data = wav_file.readframes(n_frames)

        # NumPy array conversion
        tetra_array = np.frombuffer(audio_data, dtype=np.int16)
        # Reshape to avoid interleaving channels
        tetra_array = (tetra_array.reshape(1, -1)).astype(float)
        tetra_array /= 32768. # Values between [-1,1]
        return tetra_array, frame_rate

def change_folder(file_path, new_folder_name):
    """Change the folder of a file path

    Args:
        file_path (str): File path
        folder_name (str): New folder name

    Returns:
        str: New file path
    """
    return os.path.join(new_folder_name, os.path.basename(file_path))
    

def output_wav_file(tetra_array, frame_rate, output_path):
    """Write a NumPy array representing 4 microphones to a WAV file

    Args:
        audio_array (np.ndarray): Audio data to output WARNING GIVE IT THE TRANSPOSE
        frame_rate (int): Sample rate
        output_path (str): Path to output file
    """
    with wave.open(str(output_path), 'w') as f:
        # Set parameters
        f.setnchannels(tetra_array.shape[1]) # 4 channels
        f.setsampwidth(2)  # 2 bytes for int16
        f.setframerate(frame_rate)
        # Write audio data
        audio_data = np.int16(tetra_array *32767)
        f.writeframes(audio_data.tobytes())
        
def audio_array_to_wav(audio_array : AudioArray, output_path : str):
    output_wav_file(audio_array.data_array.T, audio_array.metadata.sample_rate, output_path)

def wav_to_array(wav_path : str, use_h4 : bool, tetrahedra : Tetrahedra, snr_power : list[float] = [1.0,1.0,1.0,1.0]):
    """Convert a single wav to an AudioArray."""
    tetra_array, frame_rate = read_wav_file(wav_path)
    frequency_range = (500, 20000)
    call_type = 'whistle'
    central_frequency = 10000
    call_duration = float(tetra_array.shape[1])/frame_rate
    start_time = 0.0
    metadata = AudioMetadata(tetrahedra.id, call_type, call_duration, start_time, snr_power, frame_rate, frequency_range, central_frequency)
    audio_array = AudioArray(metadata, tetrahedra, use_h4, data_array=tetra_array)
    return audio_array

def select_plage_audio(data, sample_rate, delay : float, duration:float):
    """récupère les audios sous la forme d'un tableau (de nombre_hydrophones colonnes), 
    et renvoie les audios d'une durée duration commencant au time_start

    Args:
        data (array): data des 4 canaux
        sample_rate (int): sample rate
        time_start (array): [heures, minutes, secondes, ms]
        duration (array): [heures, minutes, secondes, ms]

    Returns:
        array: data des 4 canaux
    """
    print(f"shape data : {data.shape}")
    start_index = sample_rate*delay
    end_index = start_index + sample_rate*duration
    print(f"Starting index : {int(start_index)}")
    print(f"Ending index : {int(end_index)}")

    return data[int(start_index):int(end_index),:]
