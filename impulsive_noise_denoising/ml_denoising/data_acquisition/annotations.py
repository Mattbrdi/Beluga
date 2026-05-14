import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from impulsive_noise_denoising.stft import frequency_band, scipy_db_spectrogram, scipy_spectrogram
from impulsive_noise_denoising.ml_denoising.thresholding import threshold_model, filter_whistle_false_positives
from impulsive_noise_denoising.ml_denoising.data_acquisition.data_parser import bandpass_waveform
from impulsive_noise_denoising.wav_reader import read_wav_file


LOWCUT_HZ = 500
HIGHCUT_HZ = 20_000
SEGMENT_SECONDS = 1.0
DEFAULT_AUTO_INTERVAL_SECONDS = 0.01
WAV_EXTENSIONS = (".wav", ".wave")


def parse_args():
    parser = argparse.ArgumentParser(description="Streamlit impulsive noise mask annotation tool.")
    parser.add_argument("--inputpath", default=None, help="Folder containing preprocessed WAV files.")
    parser.add_argument("--outputpath", default=None, help="Folder where .labels.json files are saved.")
    args, _ = parser.parse_known_args()
    return args


def list_wav_files(input_folder):
    input_folder = Path(input_folder)
    if not input_folder.exists():
        return []
    return sorted(path for path in input_folder.iterdir() if path.suffix.lower() in WAV_EXTENSIONS)


def normalize_audio_data(audio_data):
    audio_data = np.asarray(audio_data)

    if audio_data.ndim == 1:
        audio_data = audio_data[np.newaxis, :]
    elif audio_data.ndim == 2:
        audio_data = audio_data.T
    else:
        raise ValueError(f"Expected mono or multichannel WAV data, got shape {audio_data.shape}")

    if np.issubdtype(audio_data.dtype, np.floating):
        return audio_data.astype(np.float64)

    if np.issubdtype(audio_data.dtype, np.integer):
        max_abs_value = max(abs(np.iinfo(audio_data.dtype).min), np.iinfo(audio_data.dtype).max)
        return audio_data.astype(np.float64) / max_abs_value

    raise ValueError(f"Unsupported WAV dtype: {audio_data.dtype}")


def read_annotation_wav(wav_path):
    try:
        return read_wav_file(wav_path)
    except ValueError as error:
        if "4 channels" not in str(error):
            raise

    sampling_rate, audio_data = wavfile.read(wav_path)
    waveform = normalize_audio_data(audio_data)
    if waveform.shape[0] not in (1, 4):
        raise ValueError(f"Expected 1 or 4 channels, got {waveform.shape[0]} channels")
    return waveform, sampling_rate


def segment_count(waveform, sampling_rate):
    samples_per_segment = int(round(SEGMENT_SECONDS * sampling_rate))
    return max(1, int(np.ceil(waveform.shape[-1] / samples_per_segment)))


def get_segment(waveform, sampling_rate, segment_index):
    samples_per_segment = int(round(SEGMENT_SECONDS * sampling_rate))
    start_sample = segment_index * samples_per_segment
    end_sample = start_sample + samples_per_segment
    segment = waveform[:, start_sample:end_sample]

    if segment.shape[-1] < samples_per_segment:
        segment = np.pad(segment, ((0, 0), (0, samples_per_segment - segment.shape[-1])))

    start_time = start_sample / sampling_rate
    end_time = end_sample / sampling_rate
    return segment, start_time, end_time


def segment_stem(wav_path, segment_start_time, segment_end_time):
    return wav_path.stem


def labels_path(output_folder, wav_path, segment_start_time, segment_end_time):
    return Path(output_folder) / f"{segment_stem(wav_path, segment_start_time, segment_end_time)}.labels.json"


def load_labels(output_folder, wav_path, segment_start_time, segment_end_time):
    path = labels_path(output_folder, wav_path, segment_start_time, segment_end_time)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return [tuple(label) for label in data.get("labels", [])]


def previous_segment_reference(wav_files, wav_index, segment_index, current_waveform, sampling_rate):
    if segment_index > 0:
        return wav_files[wav_index], segment_index - 1, current_waveform, sampling_rate

    if wav_index == 0:
        return None

    previous_wav_path = wav_files[wav_index - 1]
    previous_waveform, previous_sampling_rate = read_annotation_wav(previous_wav_path)
    previous_segment_index = segment_count(previous_waveform, previous_sampling_rate) - 1
    return previous_wav_path, previous_segment_index, previous_waveform, previous_sampling_rate


def load_previous_segment_labels(wav_files, wav_index, segment_index, waveform, sampling_rate, output_folder):
    previous_reference = previous_segment_reference(
        wav_files,
        wav_index,
        segment_index,
        waveform,
        sampling_rate,
    )
    if previous_reference is None:
        return None, []

    previous_wav_path, previous_segment_index, previous_waveform, previous_sampling_rate = previous_reference
    _, previous_start, previous_end = get_segment(
        previous_waveform,
        previous_sampling_rate,
        previous_segment_index,
    )
    labels = load_labels(output_folder, previous_wav_path, previous_start, previous_end)
    return previous_wav_path, labels


def interval_around_click(clicked_time, duration):
    half_duration = duration / 2
    start = max(0.0, clicked_time - half_duration)
    end = min(SEGMENT_SECONDS, clicked_time + half_duration)
    return float(start), float(end)


def mask_to_labels(mask, sampling_rate):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim > 1:
        mask = np.any(mask, axis=0)

    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    labels = []
    for start_idx, end_idx in changes.reshape(-1, 2):
        labels.append((start_idx / sampling_rate, end_idx / sampling_rate))
    return labels


def auto_label_segment(
    channel,
    sampling_rate,
    pulse_duration,
    pulse_overlap,
    z_threshold,
    local_pulse_radius,
):
    mask = threshold_model(
        channel,
        sampling_rate,
        pulse_duration=0.0005,
        pulse_overlap=0.00025,
        z_threshold=10,
        local_pulse_radius=10,
    )
    channel_low_freq = bandpass_waveform(channel * channel, sampling_rate, 1, 1000, 4)
    mask_low_freq = threshold_model(
        channel_low_freq,
        sampling_rate,
        pulse_duration=0.008,
        pulse_overlap=0.0025,
        z_threshold=10,
        local_pulse_radius=5,
    )
    mask = (mask.astype(bool) | mask_low_freq.astype(bool)).astype(np.uint8)

    return mask_to_labels(mask, sampling_rate)


def stable_key(*parts):
    raw_key = "|".join(str(part) for part in parts)
    return hashlib.md5(raw_key.encode("utf-8")).hexdigest()


def audio_wav_bytes(audio, sampling_rate):
    audio = np.asarray(audio, dtype=np.float32)
    buffer = io.BytesIO()
    wavfile.write(buffer, sampling_rate, audio)
    return buffer.getvalue()


def save_labels(output_folder, wav_path, sampling_rate, segment_start_time, segment_end_time, labels):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    path = labels_path(output_folder, wav_path, segment_start_time, segment_end_time)

    data = {
        "source_wav": str(wav_path),
        "segment_name": segment_stem(wav_path, segment_start_time, segment_end_time),
        "segment_start_time": float(segment_start_time),
        "segment_end_time": float(segment_end_time),
        "sampling_rate": int(sampling_rate),
        "labels": [[float(start), float(end)] for start, end in labels],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return path


def bandpass_channel(channel, sampling_rate):
    nyquist = sampling_rate / 2
    highcut = min(HIGHCUT_HZ, nyquist - 1)
    if LOWCUT_HZ >= highcut:
        return channel

    sos = butter(4, (LOWCUT_HZ, highcut), btype="bandpass", fs=sampling_rate, output="sos")
    return sosfilt(sos, channel)


def display_frequency_range(sampling_rate):
    nyquist = sampling_rate / 2
    highcut = min(HIGHCUT_HZ, nyquist - 1)
    if LOWCUT_HZ >= highcut:
        return None
    return LOWCUT_HZ, highcut


def plot_waveform(segment, sampling_rate, channel_index, labels, pending_start, pending_end):
    channel = bandpass_channel(segment[channel_index], sampling_rate)
    time_axis = np.arange(channel.size) / sampling_rate

    fig, ax = plt.subplots(figsize=(14, 4), dpi=160)
    ax.plot(time_axis, channel, linewidth=0.7, color="#1f77b4")
    decorate_annotation_axis(ax, labels, pending_start, pending_end)
    ax.set_xlim(0, SEGMENT_SECONDS)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform - channel {channel_index + 1}")
    ax.grid(True, alpha=0.2)
    return matplotlib_click_timestamp(fig, ax, key=st.session_state.click_component_key)


def plot_stft(
    segment,
    sampling_rate,
    channel_index,
    labels,
    pending_start,
    pending_end,
    is_db,
    gain_db,
    linear_scale_percent,
):
    frequency_range = display_frequency_range(sampling_rate)
    if frequency_range is None:
        st.error(f"Cannot display 2-20 kHz STFT for sampling rate {sampling_rate} Hz.")
        return None, None

    channel = segment[channel_index]
    n_fft = min(4096, channel.size)
    hop_length = max(1, n_fft // 4)

    if is_db:
        freqs, times, spectrogram = scipy_db_spectrogram(
            channel,
            sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            gain_db=gain_db,
        )
        colorbar_label = "Amplitude (dB)"
        title_prefix = "STFT dB"
    else:
        freqs, times, spectrogram = scipy_spectrogram(
            channel,
            sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        colorbar_label = "Amplitude"
        title_prefix = "STFT"

    lowcut, highcut = frequency_range
    freqs, spectrogram = frequency_band(freqs, spectrogram, lowcut, highcut)
    spectrogram_max = np.max(spectrogram)
    if is_db:
        vmax = spectrogram_max
        vmin = vmax - 80
    else:
        vmax = spectrogram_max * (linear_scale_percent / 100)
        vmin = 0

    fig, ax = plt.subplots(figsize=(14, 4), dpi=160)
    extent = [0, SEGMENT_SECONDS, freqs[0], freqs[-1]]
    im = ax.imshow(
        spectrogram,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    decorate_annotation_axis(ax, labels, pending_start, pending_end)
    ax.set_xlim(0, SEGMENT_SECONDS)
    ax.set_ylim(lowcut, highcut)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{title_prefix} 2-20 kHz - channel {channel_index + 1}")
    fig.colorbar(im, ax=ax, label=colorbar_label)
    return matplotlib_click_timestamp(fig, ax, key=st.session_state.click_component_key)


def matplotlib_click_timestamp(fig, ax, key):
    if streamlit_image_coordinates is None:
        st.pyplot(fig, clear_figure=True)
        st.warning("Install streamlit-image-coordinates to click directly on the plot.")
        plt.close(fig)
        return None

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    image = Image.fromarray(image)
    click = streamlit_image_coordinates(image, key=key)
    plt.close(fig)

    if click is None:
        return None, None

    x_pixel = click["x"]
    y_pixel = height - click["y"]
    timestamp, _ = ax.transData.inverted().transform((x_pixel, y_pixel))

    if 0 <= timestamp <= SEGMENT_SECONDS:
        return float(timestamp), f"{key}:{click['x']}:{click['y']}"
    return None, None


def install_keyboard_shortcuts():
    components.html(
        """
        <script>
        const shortcutVersion = "2026-05-12-label-toggle-v2";
        function installImpulsiveNoiseShortcuts(win) {
            if (!win || win.__impulsiveNoiseAnnotationShortcutsVersion === shortcutVersion) {
                return;
            }
            win.__impulsiveNoiseAnnotationShortcutsVersion = shortcutVersion;
            if (win.__impulsiveNoiseAnnotationShortcutsHandler) {
                win.document.removeEventListener("keydown", win.__impulsiveNoiseAnnotationShortcutsHandler, true);
            }
            win.__impulsiveNoiseAnnotationShortcutsHandler = function(event) {
                const tag = event.target.tagName;
                const isTyping = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || event.target.isContentEditable;
                if (isTyping || event.ctrlKey || event.metaKey || event.altKey) {
                    return;
                }

                const keyToButton = {
                    "s": "Set Start",
                    "e": "Set End",
                    "i": "Save Interval",
                    "v": "Toggle Labels",
                    "a": "Auto Label",
                };
                const buttonText = keyToButton[event.key.toLowerCase()];
                if (!buttonText) {
                    return;
                }

                const buttons = Array.from(window.parent.document.querySelectorAll("button"));
                const button = buttons.find((candidate) => candidate.innerText.trim().includes(buttonText));
                if (button) {
                    event.preventDefault();
                    button.click();
                }
            };
            win.document.addEventListener("keydown", win.__impulsiveNoiseAnnotationShortcutsHandler, true);
        }

        const root = window.parent;
        installImpulsiveNoiseShortcuts(root);

        for (const frame of root.document.querySelectorAll("iframe")) {
            try {
                installImpulsiveNoiseShortcuts(frame.contentWindow);
            } catch (error) {
            }
        }
        </script>
        """,
        height=0,
    )


def decorate_annotation_axis(ax, labels, pending_start, pending_end):
    for start, end in labels:
        ax.axvspan(start, end, color="#d62728", alpha=0.25)
        ax.axvline(start, color="#d62728", linewidth=1)
        ax.axvline(end, color="#d62728", linewidth=1)

    if pending_start is not None:
        ax.axvline(pending_start, color="#2ca02c", linestyle="--", linewidth=2)
    if pending_end is not None:
        ax.axvline(pending_end, color="#ff7f0e", linestyle="--", linewidth=2)


def reset_pending_points():
    st.session_state.pending_start = None
    st.session_state.pending_end = None


def delete_label(index):
    if 0 <= index < len(st.session_state.labels):
        del st.session_state.labels[index]


def toggle_label_visibility():
    st.session_state.show_labels = not st.session_state.show_labels


def update_label_interval(index, center, duration):
    duration = max(0.0, min(float(duration), SEGMENT_SECONDS))
    half_duration = duration / 2
    start = max(0.0, center - half_duration)
    end = min(SEGMENT_SECONDS, center + half_duration)

    if start == 0.0:
        end = min(SEGMENT_SECONDS, duration)
    elif end == SEGMENT_SECONDS:
        start = max(0.0, SEGMENT_SECONDS - duration)

    if 0 <= index < len(st.session_state.labels):
        st.session_state.labels[index] = (float(start), float(end))


def set_segment_state(labels):
    st.session_state.labels = labels
    st.session_state.last_click_signature = None
    st.session_state.pending_clicked_time = None
    reset_pending_points()


def init_state():
    st.session_state.setdefault("wav_index", 0)
    st.session_state.setdefault("segment_index", 0)
    st.session_state.setdefault("labels", [])
    st.session_state.setdefault("pending_start", None)
    st.session_state.setdefault("pending_end", None)
    st.session_state.setdefault("cursor_time", 0.0)
    st.session_state.setdefault("cursor_time_input", 0.0)
    st.session_state.setdefault("pending_clicked_time", None)
    st.session_state.setdefault("last_click_signature", None)
    st.session_state.setdefault("show_labels", True)


def move_to_segment(delta, wav_files, waveform, sampling_rate):
    next_segment = st.session_state.segment_index + delta
    total_segments = segment_count(waveform, sampling_rate)

    if next_segment < 0 and st.session_state.wav_index > 0:
        st.session_state.wav_index -= 1
        next_wav, next_sampling_rate = read_annotation_wav(wav_files[st.session_state.wav_index])
        st.session_state.segment_index = segment_count(next_wav, next_sampling_rate) - 1
    elif next_segment >= total_segments and st.session_state.wav_index < len(wav_files) - 1:
        st.session_state.wav_index += 1
        st.session_state.segment_index = 0
    else:
        st.session_state.segment_index = min(max(next_segment, 0), total_segments - 1)

    reset_pending_points()


def main():
    args = parse_args()
    st.set_page_config(page_title="Impulsive Noise Annotation", layout="wide")
    init_state()
    st.title("Impulsive Noise Mask Annotation")

    with st.sidebar:
        input_folder = st.text_input("Input folder", value=args.inputpath or "")
        output_folder = st.text_input("Output folder", value=args.outputpath or "")

    if not input_folder or not output_folder:
        st.info("Provide --inputpath and --outputpath, or fill both folders in the sidebar.")
        return

    wav_files = list_wav_files(input_folder)
    if not wav_files:
        st.warning("No WAV files found in the input folder.")
        return

    st.session_state.wav_index = min(st.session_state.wav_index, len(wav_files) - 1)
    wav_path = wav_files[st.session_state.wav_index]
    waveform, sampling_rate = read_annotation_wav(wav_path)

    total_segments = segment_count(waveform, sampling_rate)
    st.session_state.segment_index = min(st.session_state.segment_index, total_segments - 1)

    segment, segment_start, segment_end = get_segment(
        waveform,
        sampling_rate,
        st.session_state.segment_index,
    )
    current_labels = load_labels(output_folder, wav_path, segment_start, segment_end)
    current_key = f"{wav_path}_{st.session_state.segment_index}"
    if st.session_state.get("current_key") != current_key:
        st.session_state.current_key = current_key
        set_segment_state(current_labels)

    with st.sidebar:
        st.write(f"File {st.session_state.wav_index + 1} / {len(wav_files)}")
        selected_wav = st.selectbox(
            "WAV file",
            options=list(range(len(wav_files))),
            format_func=lambda index: wav_files[index].name,
            index=st.session_state.wav_index,
        )
        if selected_wav != st.session_state.wav_index:
            st.session_state.wav_index = selected_wav
            st.session_state.segment_index = 0
            reset_pending_points()
            st.rerun()

        selected_segment = st.number_input(
            "Segment",
            min_value=1,
            max_value=total_segments,
            value=st.session_state.segment_index + 1,
            step=1,
        )
        if selected_segment - 1 != st.session_state.segment_index:
            st.session_state.segment_index = selected_segment - 1
            reset_pending_points()
            st.rerun()

        view_mode = st.radio("View", ["Waveform", "STFT"], horizontal=True)
        channel_options = list(range(waveform.shape[0]))
        channel_index = st.selectbox(
            "Channel",
            options=channel_options,
            format_func=lambda i: f"Channel {i + 1}",
        )
        stft_gain_db = st.number_input(
            "STFT gain (dB)",
            value=0.0,
            step=5.0,
            format="%.1f",
        )
        stft_is_db = st.toggle("STFT dB scale", value=True)
        stft_linear_scale_percent = st.number_input(
            "Linear STFT scale (% max)",
            min_value=1.0,
            max_value=100.0,
            value=100.0,
            step=5.0,
            format="%.1f",
            disabled=stft_is_db,
        )
        auto_interval_on_click = st.toggle("Auto interval on click", value=False)
        auto_interval_duration = st.slider(
            "Auto interval duration (s)",
            min_value=0.0,
            max_value=0.05,
            value=DEFAULT_AUTO_INTERVAL_SECONDS,
            step=0.001,
            disabled=not auto_interval_on_click,
        )
        st.divider()
        st.write("Threshold auto-label")
        threshold_pulse_duration = st.number_input(
            "Pulse duration (s)",
            value=0.005,
            step=0.001,
            format="%.6f",
        )
        threshold_pulse_overlap = st.number_input(
            "Pulse overlap (s)",
            value=0.0025,
            step=0.0005,
            format="%.6f",
        )
        threshold_z_threshold = st.number_input(
            "Z threshold",
            value=3.0,
            step=0.5,
            format="%.3f",
        )
        threshold_local_pulse_radius = st.number_input(
            "Local pulse radius",
            value=10,
            step=1,
        )

    st.session_state.click_component_key = stable_key(
        "plot_click",
        current_key,
        view_mode,
        channel_index,
    )

    st.caption(
        f"{wav_path.name} | segment {st.session_state.segment_index + 1}/{total_segments} | "
        f"{segment_start:.3f}s - {segment_end:.3f}s | {sampling_rate} Hz"
    )

    if st.session_state.pending_clicked_time is not None:
        st.session_state.cursor_time_input = st.session_state.pending_clicked_time
        st.session_state.cursor_time = st.session_state.pending_clicked_time
        st.session_state.pending_clicked_time = None

    st.session_state.cursor_time_input = min(max(st.session_state.cursor_time_input, 0.0), SEGMENT_SECONDS)
    st.session_state.cursor_time = st.session_state.cursor_time_input
    controls, label_panel = st.columns([3, 2])
    with controls:
        cursor_step = 1.0 / sampling_rate
        st.number_input(
            "Cursor time inside segment (s)",
            min_value=0.0,
            max_value=SEGMENT_SECONDS,
            step=cursor_step,
            format="%.6f",
            key="cursor_time_input",
        )
        st.session_state.cursor_time = st.session_state.cursor_time_input
        start_col, end_col, save_col, clear_col, labels_col, auto_col = st.columns(6)
        if start_col.button("Set Start"):
            st.session_state.pending_start = st.session_state.cursor_time
        if end_col.button("Set End"):
            st.session_state.pending_end = st.session_state.cursor_time
        if save_col.button("Save Interval"):
            start = st.session_state.pending_start
            end = st.session_state.pending_end
            if start is None or end is None:
                st.warning("Set both start and end before saving the interval.")
            elif start == end:
                st.warning("Start and end cannot be identical.")
            else:
                label = tuple(sorted((float(start), float(end))))
                st.session_state.labels.append(label)
                reset_pending_points()
        if clear_col.button("Clear Pending"):
            reset_pending_points()
        labels_col.button(
            "Toggle Labels",
            on_click=toggle_label_visibility,
        )
        if auto_col.button("Auto Label"):
            try:
                detected_labels = auto_label_segment(
                    segment[channel_index],
                    sampling_rate,
                    threshold_pulse_duration,
                    threshold_pulse_overlap,
                    threshold_z_threshold,
                    threshold_local_pulse_radius,
                )
                if detected_labels:
                    st.session_state.labels.extend(detected_labels)
                    reset_pending_points()
                    st.success(f"Added {len(detected_labels)} automatic label(s).")
                else:
                    st.info("No automatic labels detected.")
            except ValueError as error:
                st.error(f"Auto Label failed: {error}")

        label_visibility = "shown" if st.session_state.show_labels else "hidden"
        st.caption(
            "Shortcuts: s = set start, e = set end, i = save interval, "
            f"v = toggle labels ({label_visibility}), a = auto label"
        )
        if auto_interval_on_click:
            st.caption(f"Auto interval click mode: +/- {auto_interval_duration / 2:.6f}s")
        install_keyboard_shortcuts()

        if st.button("Apply Previous Labels"):
            previous_wav_path, previous_labels = load_previous_segment_labels(
                wav_files,
                st.session_state.wav_index,
                st.session_state.segment_index,
                waveform,
                sampling_rate,
                output_folder,
            )
            if previous_wav_path is None:
                st.warning("There is no previous segment to copy from.")
            elif not previous_labels:
                st.warning("The previous segment has no saved labels.")
            else:
                st.session_state.labels = list(previous_labels)
                reset_pending_points()

        audio_col, time_col = st.columns([1, 2])
        listen_current_audio = audio_col.button("Listen Current Audio")
        time_col.write(
            f"{segment_start:.9f}s - {segment_end:.9f}s "
            f"| cursor {segment_start + st.session_state.cursor_time:.9f}s"
        )
        if listen_current_audio:
            current_audio = segment[channel_index]
            st.audio(
                audio_wav_bytes(current_audio, sampling_rate),
                format="audio/wav",
            )

    with label_panel:
        st.write("Labels for current segment")
        if not st.session_state.labels:
            st.caption("No impulsive noise intervals yet.")
        for index, (start, end) in enumerate(st.session_state.labels):
            row = st.columns([3, 1])
            row[0].write(f"{index + 1}. {start:.6f}s - {end:.6f}s")
            row[1].button(
                "Delete",
                key=f"delete_{current_key}_{index}",
                on_click=delete_label,
                args=(index,),
            )
            current_center = (start + end) / 2
            current_duration = end - start
            size_key = f"label_size_{current_key}_{index}"
            shift_key = f"label_center_shift_{current_key}_{index}"
            previous_shift_key = f"label_center_shift_previous_{current_key}_{index}"

            size_slider_max = max(0.05, current_duration, 0.001)
            refined_duration = st.slider(
                f"Size {index + 1} (s)",
                min_value=0.0,
                max_value=float(min(SEGMENT_SECONDS, size_slider_max)),
                value=float(min(current_duration, min(SEGMENT_SECONDS, size_slider_max))),
                step=0.0005,
                format="%.6f",
                key=size_key,
            )

            center_shift_ms = st.slider(
                f"Center shift {index + 1} (ms)",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                format="%.1f",
                key=shift_key,
            )
            previous_shift_ms = st.session_state.get(previous_shift_key, 0.0)
            shift_delta_seconds = (center_shift_ms - previous_shift_ms) / 1000
            st.session_state[previous_shift_key] = center_shift_ms

            if refined_duration != current_duration or shift_delta_seconds != 0:
                update_label_interval(
                    index,
                    current_center + shift_delta_seconds,
                    refined_duration,
                )

    labels_for_plot = st.session_state.labels if st.session_state.show_labels else []

    if view_mode == "Waveform":
        clicked_time, click_signature = plot_waveform(
            segment,
            sampling_rate,
            channel_index,
            labels_for_plot,
            st.session_state.pending_start,
            st.session_state.pending_end,
        )
    else:
        clicked_time, click_signature = plot_stft(
            segment,
            sampling_rate,
            channel_index,
            labels_for_plot,
            st.session_state.pending_start,
            st.session_state.pending_end,
            stft_is_db,
            stft_gain_db,
            stft_linear_scale_percent,
        )

    if clicked_time is not None and click_signature != st.session_state.get("last_click_signature"):
        st.session_state.last_click_signature = click_signature
        st.session_state.pending_clicked_time = round(clicked_time, 6)
        if auto_interval_on_click:
            st.session_state.labels.append(interval_around_click(clicked_time, auto_interval_duration))
            reset_pending_points()
        st.rerun()

    if st.session_state.get("last_click_signature"):
        st.success(f"Clicked timestamp: {st.session_state.cursor_time:.6f}s")

    nav_col, save_file_col, status_col = st.columns([2, 2, 3])
    with nav_col:
        previous_col, next_col = st.columns(2)
        previous_col.button(
            "Previous",
            on_click=move_to_segment,
            args=(-1, wav_files, waveform, sampling_rate),
        )
        next_col.button(
            "Next",
            on_click=move_to_segment,
            args=(1, wav_files, waveform, sampling_rate),
        )

    with save_file_col:
        if st.button("Save Labels File"):
            path = save_labels(
                output_folder,
                wav_path,
                sampling_rate,
                segment_start,
                segment_end,
                st.session_state.labels,
            )
            st.success(f"Saved {path.name}")

    with status_col:
        path = labels_path(output_folder, wav_path, segment_start, segment_end)
        if path.exists():
            st.caption(f"Loaded labels file: {path.name}")
        else:
            st.caption("No labels file saved for this segment yet.")


if __name__ == "__main__":
    main()
