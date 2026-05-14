import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from impulsive_noise_denoising.wav_reader import read_wav_file


LOWCUT_HZ = 500
HIGHCUT_HZ = 20_000
SEGMENT_DURATION_SECONDS = 1.0


def bandpass_waveform(waveform, sampling_rate, lowcut=LOWCUT_HZ, highcut=HIGHCUT_HZ, order=4):
    nyquist = sampling_rate / 2
    if highcut >= nyquist:
        raise ValueError(
            f"Cannot apply {lowcut}-{highcut} Hz bandpass with sampling rate {sampling_rate} Hz"
        )
    if lowcut >= highcut:
        raise ValueError(
            f"Invalid bandpass range {lowcut}-{highcut} Hz for sampling rate {sampling_rate} Hz"
        )

    sos = butter(order, (lowcut, highcut), btype="bandpass", fs=sampling_rate, output="sos")
    return sosfilt(sos, waveform, axis=-1)


def normalize_waveform(waveform):
    scale = np.percentile(np.abs(waveform), 95)
    if scale == 0:
        return np.zeros_like(waveform, dtype=np.float32)

    waveform = waveform / scale
    waveform = np.clip(waveform, -5, 5)
    waveform = waveform / 5
    return waveform.astype(np.float32)


def preprocess_waveform(waveform, sampling_rate):
    waveform = bandpass_waveform(waveform, sampling_rate)
    return normalize_waveform(waveform)


def segment_waveform(waveform, sampling_rate, segment_duration=SEGMENT_DURATION_SECONDS):
    segment_size = int(round(segment_duration * sampling_rate))
    if segment_size <= 0:
        raise ValueError("segment_duration must produce at least one sample")

    total_samples = waveform.shape[-1]
    number_of_segments = max(1, int(np.ceil(total_samples / segment_size)))

    for segment_index in range(number_of_segments):
        start_sample = segment_index * segment_size
        end_sample = start_sample + segment_size
        segment = waveform[..., start_sample:end_sample]

        if segment.shape[-1] < segment_size:
            pad_width = [(0, 0)] * segment.ndim
            pad_width[-1] = (0, segment_size - segment.shape[-1])
            segment = np.pad(segment, pad_width, mode="constant")

        start_time = start_sample / sampling_rate
        yield segment, start_time


def timestamp_name(start_time):
    milliseconds = int(round(start_time * 1000))
    return f"{milliseconds:010d}ms"


def save_waveform(path, waveform, sampling_rate):
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.T
    wavfile.write(path, sampling_rate, waveform)


def parse_wav_file(inputpath, outputpath, split):
    inputpath = Path(inputpath)
    outputpath = Path(outputpath)
    outputpath.mkdir(parents=True, exist_ok=True)

    waveform, sampling_rate = read_wav_file(inputpath)

    saved_paths = []
    for segment, start_time in segment_waveform(waveform, sampling_rate):
        processed_segment = preprocess_waveform(segment, sampling_rate)
        if split:
            for i, channel in enumerate(processed_segment):
                output_file = outputpath / f"{inputpath.stem}_{timestamp_name(start_time)}_channel_{i}.wav"
                save_waveform(output_file, channel, sampling_rate)
                saved_paths.append(output_file)
        else:
            output_file = outputpath / f"{inputpath.stem}_{timestamp_name(start_time)}.wav"
            save_waveform(output_file, processed_segment, sampling_rate)
            saved_paths.append(output_file)
    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a WAV file into 1-second preprocessed waveform segments."
    )
    parser.add_argument("--inputpath", required=True, help="Path to the input WAV file.")
    parser.add_argument("--outputpath", required=True, help="Folder where output WAV segments are saved.")
    parser.add_argument(
        "--split",
        action="store_true",
        help="Save each channel as a separate mono WAV instead of one multichannel WAV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    saved_paths = parse_wav_file(args.inputpath, args.outputpath, args.split)
    print(f"Saved {len(saved_paths)} waveform segments to {args.outputpath}")


if __name__ == "__main__":
    main()
