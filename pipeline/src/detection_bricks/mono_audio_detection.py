import torch
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import warnings
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
import numpy as np
import librosa
import os
import json
import time
from skimage.transform import resize
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

class MobileNetMultilabel(nn.Module):
    def __init__(self, num_classes, pretrained=True, dense_layer_size=256, max_feature_layer=None):
        super(MobileNetMultilabel, self).__init__()

        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.to(self.device)

        print(f"Loading MobileNetV3_Small model on {self.device}")
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_small(weights=weights)

        # Modify first conv layer for single-channel input
        mobilenet.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

        print(f"Number of feature layers: {len(mobilenet.features)}")
        print(f"Using up to layer {max_feature_layer}")

        # Truncate MobileNet features if requested
        if max_feature_layer is not None:
            if not (0 <= max_feature_layer <= 12):
                raise ValueError("max_feature_layer must be between 0 and 12")
            self.features = nn.Sequential(*mobilenet.features[:max_feature_layer + 1])

            # Output feature size map (empirically determined)
            feature_size_by_layer = {
                0: 16, 1: 16, 2: 24, 3: 24, 4: 40, 5: 40,
                6: 40, 7: 48, 8: 48, 9: 96, 10: 96, 11: 96, 12: 576
            }
            num_features = feature_size_by_layer[max_feature_layer]
        else:
            self.features = mobilenet.features
            num_features = mobilenet.classifier[0].in_features  # usually 576

        print(f"Feature size after backbone: {num_features}")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Final multilabel classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features, dense_layer_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dense_layer_size, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        output = self.classifier(features)
        return output  # raw logits for BCEWithLogitsLoss

    def save_model(self, file_path):
        torch.save({
            'state_dict': self.state_dict(),
        }, file_path)
        print(f"Model weights saved to {file_path}")

    def _rebuild(self, num_classes: int, dense_layer_size: int, pretrained: bool, max_feature_layer: int | None):
        """Reconstruit features/avgpool/classifier selon les paramètres demandés."""
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_small(weights=weights)

        # 1-channel input
        mobilenet.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

        if max_feature_layer is not None:
            if not (0 <= max_feature_layer <= 12):
                raise ValueError("max_feature_layer must be between 0 and 12")
            self.features = nn.Sequential(*mobilenet.features[:max_feature_layer + 1])

            feature_size_by_layer = {
                0: 16, 1: 16, 2: 24, 3: 24, 4: 40, 5: 40,
                6: 40, 7: 48, 8: 48, 9: 96, 10: 96, 11: 96, 12: 576
            }
            num_features = feature_size_by_layer[max_feature_layer]
        else:
            self.features = mobilenet.features
            num_features = mobilenet.classifier[0].in_features  # usually 576

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(num_features, dense_layer_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dense_layer_size, num_classes)
        )


    def load_model(self, file_path):
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=True)
        sd = checkpoint["state_dict"]

        # --- (A) Déduire max_feature_layer depuis les clés du state_dict ---
        feat_ids = []
        for k in sd.keys():
            if k.startswith("features."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    feat_ids.append(int(parts[1]))
        max_feature_layer_sd = max(feat_ids) if feat_ids else None  # ex: 7 ou 8 (QAT), 12 (full)

        # --- (B) Déduire tailles du classifier depuis les poids ---
        # classifier = [Linear(num_features->dense), ReLU, Dropout, Linear(dense->num_classes)]
        dense_layer_size = sd["classifier.0.weight"].shape[0]      # 256
        num_features      = sd["classifier.0.weight"].shape[1]     # 48 ou 576 (info)
        num_classes       = sd["classifier.3.weight"].shape[0]     # nb classes

        # --- (C) Récupérer metadata (si présent) ---
        pretrained_ckpt = bool(checkpoint.get("pretrained", True))
        # qat = bool(checkpoint.get("qat", False))  # si tu veux le stocker pour info

        # --- (D) Reconstruire l'arch qui match le checkpoint ---
        # (on se fie au max_feature_layer réel du state_dict => le plus fiable)
        self._rebuild(
            num_classes=num_classes,
            dense_layer_size=int(dense_layer_size),
            pretrained=pretrained_ckpt,
            max_feature_layer=max_feature_layer_sd
        )
        self.to(self.device)

        # Petite vérif utile (temporaire)
        # print("Rebuilt max_feature_layer =", max_feature_layer_sd, "classifier in_features =", self.classifier[0].in_features, "(ckpt:", num_features, ")")

        # --- (E) Charger les poids ---
        self.load_state_dict(sd, strict=True)
        print("Loaded OK | max_feature_layer=", max_feature_layer_sd,
            "| classifier:", self.classifier[0].in_features, "->", self.classifier[-1].out_features)

        self.eval()

        print(
            f"Model weights loaded from {file_path} | "
            f"max_feature_layer={max_feature_layer_sd} | "
            f"in_features={self.classifier[0].in_features}"
        )

      
class SpectrogramGenerator:
    def __init__(
            self,
            n_mels=64,
            n_fft=1024,
            hop_length=128,
            fmin=200,
            sample_rate=192000, 
            power_to_db_ref=1e-12
    ):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.sample_rate = sample_rate
        self.power_to_db_ref = power_to_db_ref

    def compute_mel_power_spect(self, audio):
        if not isinstance(audio, np.ndarray):
            raise TypeError("Audio input must be a numpy array.")

        stft_complex = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        stft_magnitude = np.abs(stft_complex)

        # Compute Mel spectrogram
        mel_power_spect = librosa.feature.melspectrogram(
            S=stft_magnitude**2,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
        )
        return mel_power_spect
    
    def apply_hydrophone_sensitivity_power(self, power_spect, hydrophone_sensitivity):
        # Convert system sensitivity from dB to linear scale
        sensitivity_linear = 10 ** (hydrophone_sensitivity / 20)

        # Apply calibration to convert to pressure units (μPa)
        calibrated_power_spect = power_spect / (sensitivity_linear ** 2)

        return calibrated_power_spect
    
    def power_to_db(self, spect, hydrophone_sensitivity=None):
        if not isinstance(spect, np.ndarray):
            raise TypeError("Spectrogram input must be a numpy array.")
        
        if hydrophone_sensitivity is None:
            return librosa.power_to_db(spect, ref=self.power_to_db_ref)
        else :
            calibrated_power_spect = self.apply_hydrophone_sensitivity_power(spect, hydrophone_sensitivity)
            spect_dB = librosa.power_to_db(calibrated_power_spect, ref=1.0)
            freq_resolution = self.sample_rate / self.n_fft

            spect_dB = spect_dB - 10 * np.log10(freq_resolution)
            
            return spect_dB
    
    def min_max_normalization(self, spect):
        if not isinstance(spect, np.ndarray):
            raise TypeError("Spectrogram input must be a numpy array.")
        if spect.min() == spect.max():
            # self.plot_spect(spect)
            # print("here")
            return np.zeros_like(spect)
        return (spect - spect.min()) / (spect.max() - spect.min())
    
    def resize_spect(self, spect, img_shape):
        if not isinstance(spect, np.ndarray):
            raise TypeError("Spectrogram input must be a numpy array.")
        if len(img_shape) != 2:
            raise ValueError("img_shape must be a tuple of (height, width).")
        return resize(spect, img_shape, mode="reflect", anti_aliasing=True)
    
    def get_mel_frequencies(self):
        return librosa.mel_frequencies(n_mels=self.n_mels, fmin=self.fmin, fmax=self.sample_rate // 2)
    
    def compute_psd(self, audio, hydrophone_sensitivity=None):
        stft_complex = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        stft_magnitude = np.abs(stft_complex)
        power_spect = stft_magnitude ** 2

        # Apply calibration if needed
        if hydrophone_sensitivity is not None:
            # print(f"Applying hydrophone sensitivity: {hydrophone_sensitivity} dB re 1V/uPa")
            power_spect = self.apply_hydrophone_sensitivity_power(power_spect, hydrophone_sensitivity)

        # Average power over time (axis=1)
        avg_power_per_freq = np.mean(power_spect, axis=1)

        # Frequency bin width
        freq_resolution = self.sample_rate / self.n_fft

        # Convert to PSD in dB re 1 μPa²/Hz
        psd_db = 10 * np.log10(avg_power_per_freq / freq_resolution)
        psd_db = psd_db - 10 * np.log10(freq_resolution)

        # Frequency axis
        freqs = np.linspace(0, self.sample_rate / 2, len(avg_power_per_freq))

        return freqs, psd_db
    



    
    def denoise_spect(self, spect, noise_profile, method="gating", gating_percentage_threshold=0.1):
        
        # Validate inputs
        if not isinstance(spect, np.ndarray) or not isinstance(noise_profile, np.ndarray):
            raise TypeError("Both spect and noise_profile must be numpy arrays.")
        
        # Ensure noise_profile has the correct shape (64, 1)
        if noise_profile.shape == (spect.shape[0],):
            noise_profile = noise_profile[:, np.newaxis]
        elif noise_profile.shape != (spect.shape[0], 1):
            raise ValueError(f"Noise profile must have shape {(spect.shape[0], 1)}. Got {noise_profile.shape}")

        if method not in ["gating", "subtraction"]:
            raise ValueError("Invalid method. Choose either 'gating' or 'subtraction'.")

        if method == "gating":

            threshold_per_freq_band = noise_profile * gating_percentage_threshold

            #TODO just to test rename if I keep like this
            # threshold_per_freq_band = gating_percentage_threshold
           

            spect = np.where(
                spect > (noise_profile + threshold_per_freq_band),
                spect,
                0
                # spect.min() if spect.min() >= 0 else 0
            )

        else:
            spect = np.maximum(spect - noise_profile, 0)
            
        return spect
    
    def compute_noise_profile(self, spect, method="median", quantile=None):

        if method == "median":
            return np.median(spect, axis=1, keepdims=True)

        elif method == "max":
            return np.max(spect, axis=1, keepdims=True)

        elif method == "quantile":
            if quantile is None or not (0 <= quantile <= 1):
                raise ValueError("For method='quantile', you must provide a quantile between 0 and 1.")
            return np.quantile(spect, q=quantile, axis=1, keepdims=True)

        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'median', 'max', or 'quantile'.")

    
    
    
    def plot_spect(self, spect, title="Mel Spectrogram", figsize=(10, 4), rois_df=None, vmin=None, vmax=None, save_path=None, show=True, ax=None):
        """
        Plot the Mel spectrogram and optionally save it to disk.

        Args:
            spect (np.ndarray): The spectrogram data.
            title (str, optional): Title for the plot. Defaults to "Mel Spectrogram".
            figsize (tuple, optional): Figure size. Defaults to (10, 4).
            rois_df (pd.DataFrame, optional): DataFrame containing ROIs to plot. Defaults to None.
            vmin (float, optional): Minimum value for the colorbar range. Defaults to None (autoscale).
            vmax (float, optional): Maximum value for the colorbar range. Defaults to None (autoscale).
            save_path (str, optional): Path to save the figure. If None, the figure is not saved. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to True.
        """
        if not isinstance(spect, np.ndarray):
            raise TypeError("Spectrogram input must be a numpy array.")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        # Pass vmin and vmax to specshow
        img = librosa.display.specshow(
            spect,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            y_axis="mel",
            x_axis='time',
            ax=ax,
            vmin=vmin, # Add vmin
            vmax=vmax  # Add vmax
        )
        # The colorbar will now respect the vmin/vmax set in specshow
        fig.colorbar(img, ax=ax, format='%+2.0f dB') # Added format for clarity, adjust as needed

        if spect.shape[0] != self.n_mels:
            print(f"Removing the time and freq axis because of resized spectrogram with shape {spect.shape}")
            ax.axis('off')

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Mel)")

        # Plot ROIs if provided
        if rois_df is not None:
            for _, roi in rois_df.iterrows():
                # color = 'cyan' if roi["source"] == "hf_calls" else 'red'
                color = 'cyan' 

                # Convert Timedelta to seconds if necessary
                x = roi["min_x_s"].total_seconds() if isinstance(roi["min_x_s"], pd.Timedelta) else roi["min_x_s"]
                w = (roi["max_x_s"].total_seconds() if isinstance(roi["max_x_s"], pd.Timedelta) else roi["max_x_s"]) - x

                y = roi["min_freq_hz"]
                h = roi["max_freq_hz"] - roi["min_freq_hz"]

                rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

        # Save the figure if a path is provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to {save_path}")

        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close(fig)

    def load_audio(self, audio_path):
        if not isinstance(audio_path, str):
            raise TypeError("Audio path must be a string.")
        try:
            # Get the original sample rate without loading the audio
            original_sr = librosa.get_samplerate(audio_path)
            
            # Load and resample the audio to the target sample rate
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
            
            # Return the audio data, target sample rate, and original sample rate
            return audio, self.sample_rate, original_sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")
        
    '''
    def load_audio(self, audio_path, start_time, duration):
        if not isinstance(audio_path, str):
            raise TypeError("Audio path must be a string.")
        try:
            # Get the original sample rate without loading the audio
            original_sr = librosa.get_samplerate(audio_path)
            
            # Load and resample the audio to the target sample rate
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, offset=start_time, duration=duration)
            #audio, _ = librosa.load(audio_path, sr=self.sample_rate)
            
            # Return the audio data, target sample rate, and original sample rate
            return audio, self.sample_rate, original_sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")'''
        
    def plot_mel_band_intensity(self, spect, target_freq_hz, title=None, noise_profile=None, gating_percentage_threshold=0):
        """
        Plot the intensity over time of a specific frequency (in Hz) by converting it to the closest Mel band.
        Optionally overlays the noise profile value for that frequency band.
        """
        if not isinstance(spect, np.ndarray):
            raise TypeError("Spectrogram input must be a numpy array.")
        if not (0 < target_freq_hz < self.sample_rate // 2):
            raise ValueError(f"Frequency must be between 0 and Nyquist frequency ({self.sample_rate // 2} Hz).")

        mel_frequencies = self.get_mel_frequencies()
        mel_band_idx = np.argmin(np.abs(mel_frequencies - target_freq_hz))
        closest_freq = mel_frequencies[mel_band_idx]

        band_intensity = spect[mel_band_idx, :]

        plt.figure(figsize=(5, 2.5))
        plt.plot(band_intensity, label=f"Mel Band {mel_band_idx} ~ {closest_freq:.1f} Hz")
        
        if noise_profile is not None:
            if not isinstance(noise_profile, np.ndarray):
                raise TypeError("Noise profile must be a numpy array.")
            if noise_profile.shape == (spect.shape[0],):
                noise_value = noise_profile[mel_band_idx]
            elif noise_profile.shape == (spect.shape[0], 1):
                noise_value = noise_profile[mel_band_idx, 0]
            else:
                raise ValueError(f"Noise profile must have shape ({spect.shape[0]},) or ({spect.shape[0]}, 1). Got {noise_profile.shape}")

            plt.axhline(y=noise_value + noise_value*gating_percentage_threshold, color='red', linestyle='--', label="Noise Level")

        plt.xlabel("Time Frame")
        plt.ylabel("Intensity (dB or Linear)")
        plt.title(title or f"Intensity Over Time for ~{closest_freq:.1f} Hz")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _filter_by_frequency_range(self, spect, min_freq_hz=None, max_freq_hz=None):
        """
        Helper method to filter a spectrogram by frequency range.
        
        Args:
            spect (np.ndarray): Input spectrogram
            min_freq_hz (float, optional): Minimum frequency in Hz. Defaults to None.
            max_freq_hz (float, optional): Maximum frequency in Hz. Defaults to None.
            
        Returns:
            tuple: (filtered_spect, freq_indices) - The filtered spectrogram and the indices used for filtering
        """
        # If no frequency bounds are specified, use the entire spectrogram
        if min_freq_hz is None and max_freq_hz is None:
            return spect, np.arange(spect.shape[0])
        
        # Select frequency bands based on min_freq_hz and max_freq_hz
        mel_frequencies = self.get_mel_frequencies()
        freq_mask = np.ones_like(mel_frequencies, dtype=bool)
        if min_freq_hz is not None:
            freq_mask &= (mel_frequencies >= min_freq_hz)
        if max_freq_hz is not None:
            freq_mask &= (mel_frequencies <= max_freq_hz)
            
        freq_indices = np.where(freq_mask)[0]
        
        if len(freq_indices) == 0:
            raise ValueError(f"No frequency bands found between {min_freq_hz}Hz and {max_freq_hz}Hz")
            
        # Filter spectrogram to the selected frequency bands
        filtered_spect = spect[freq_indices, :]
        
        return filtered_spect, freq_indices
    
    def compute_db_in_freq_range(self, spect, min_freq_hz=None, max_freq_hz=None):
        """
        Compute dB statistics (mean, max, min, median) between specified frequency bounds.
        If no bounds are specified, computes statistics for the entire spectrogram.
        
        Args:
            spect (np.ndarray): Input spectrogram in dB
            min_freq_hz (float, optional): Minimum frequency in Hz
            max_freq_hz (float, optional): Maximum frequency in Hz
            
        Returns:
            dict: Dictionary containing mean, max, min, and median dB values
        """
        if not isinstance(spect, np.ndarray):
            raise TypeError("Spectrogram input must be a numpy array.")
            
        # Filter spectrogram by frequency range
        freq_band_data, _ = self._filter_by_frequency_range(spect, min_freq_hz, max_freq_hz)
        
        # Compute statistics
        stats = {
            'mean': round(float(np.mean(freq_band_data)), 1),
            'max': round(float(np.max(freq_band_data)), 1),
            'min': round(float(np.min(freq_band_data)), 1),
            'median': round(float(np.median(freq_band_data)), 1)
        }
        
        return stats
    
    def compute_peak_snr_with_time_compressed(self, signal_spect_db, noise_spect_db, min_freq_hz=None, max_freq_hz=None):
        """
        Since we are predicting calls in a fixed window, the signal can be across the whole window, or only in a part of it.
        So we compress the time axis taking the 95th percentile to make sure we get the signal intensity.

        SNR_dB = Signal_dB - Noise_dB

        Args:
            signal_spect_db (np.ndarray): Spectrogram with signal + noise (in dB units).
            noise_spect_db (np.ndarray): Spectrogram with only noise (reference, in dB units).
            min_freq_hz (float, optional): Minimum frequency in Hz for SNR calculation. Defaults to None (use all frequencies).
            max_freq_hz (float, optional): Maximum frequency in Hz for SNR calculation. Defaults to None (use all frequencies).

        Returns:
            snr_db_per_band (np.ndarray): SNR per mel band within the specified range (in dB, array of shape (n_selected_mels,)).
            snr_stats (dict): Dictionary containing mean, max, min, and median SNR (in dB) values over the selected bands.
        """
        if not isinstance(signal_spect_db, np.ndarray) or not isinstance(noise_spect_db, np.ndarray):
            raise TypeError("Both signal_spect_db and noise_spect_db must be numpy arrays.")
        
        if signal_spect_db.shape[0] != noise_spect_db.shape[0]:
            print(f"signal_spect_db.shape: {signal_spect_db.shape}")
            print(f"noise_spect_db.shape: {noise_spect_db.shape}")
            raise ValueError(f"Spectrograms must have the same number of frequency bands. Got {signal_spect_db.shape[0]} and {noise_spect_db.shape[0]} bands.")

        # Filter spectrograms by frequency range
        signal_spect_db_filtered, _ = self._filter_by_frequency_range(signal_spect_db, min_freq_hz, max_freq_hz)
        noise_spect_db_filtered, _ = self._filter_by_frequency_range(noise_spect_db, min_freq_hz, max_freq_hz)


        # noise_db_per_band = np.median(noise_spect_db_filtered, axis=1)
        noise_db_per_band = np.quantile(noise_spect_db_filtered, q=0.95, axis=1)    # Shape: (n_selected_mels,)

        # Calculate 95th percentile and each 10th percentile
        signal_db_per_band_95th = np.quantile(signal_spect_db_filtered, q=0.95, axis=1)  # Shape: (n_selected_mels,)
        snr_db_per_band_95th = signal_db_per_band_95th - noise_db_per_band

        # Calculate for each 10th percentile (0.1 to 0.9)
        snr_stats = {'95th': round(float(np.max(snr_db_per_band_95th)), 1)}
        
        for i in range(1, 10):
            q = i / 10
            signal_db_per_band = np.quantile(signal_spect_db_filtered, q=q, axis=1)
            snr_db_per_band = signal_db_per_band - noise_db_per_band
            
            percentile_name = f'{int(q*100)}th'
            snr_stats[percentile_name] = round(float(np.max(snr_db_per_band)), 1)




        # binary_mask = signal_spect_db_filtered > (noise_db_per_band[:, np.newaxis] + 5)
        # active_time_fraction = np.mean(np.any(binary_mask, axis=0))

        # snr_stats['active_time_fraction_1'] = round(active_time_fraction, 2)

        snr_db = signal_spect_db_filtered - noise_db_per_band[:, np.newaxis]
        # snr_db_per_time = np.quantile(snr_db, q=0.95, axis=0)
        snr_db_per_time = np.max(snr_db, axis=0)
        snr_time_mask = snr_db_per_time > 5
        active_time_fraction = np.mean(snr_time_mask)
        # Calculate average SNR across all time points
        avg_snr = np.mean(snr_db_per_time)
        snr_stats['avg_snr'] = round(avg_snr, 2)
        # Calculate average SNR only for time points where SNR > 5
        avg_snr_above_5 = np.mean(snr_db_per_time[snr_time_mask]) if np.any(snr_time_mask) else 0
        snr_stats['avg_snr_above_5'] = round(avg_snr_above_5, 2)


        snr_stats['active_time_fraction_1'] = active_time_fraction



        # # Calculate coverage ratio and estimated duration
        # percentiles = np.arange(10, 100, 10)  # 10, 20, 30, ..., 90
        # snr_above_thresh = []

        # for p in percentiles:
        #     q = p / 100
        #     signal_db_per_band = np.quantile(signal_spect_db_filtered, q=q, axis=1)
        #     snr_db_per_band = signal_db_per_band - noise_db_per_band
        #     max_snr = float(np.max(snr_db_per_band))
        #     snr_above_thresh.append(max_snr > 5)

        # coverage_ratio = np.mean(snr_above_thresh)  # Ratio of time with SNR > 5
        # # estimated_duration = round(coverage_ratio * 1, 3)

        
        # snr_stats['active_time_fraction_2'] = round(coverage_ratio, 2)

       


        return snr_stats

    
    def compute_snr_through_time(
        self,
        signal_spect_db,
        noise_spect_db,
        min_freq_hz=None,
        max_freq_hz=None,
        snr_threshold=5,
        time_threshold=0.05
    ):
        # Validate input types
        if not isinstance(signal_spect_db, np.ndarray) or not isinstance(noise_spect_db, np.ndarray):
            raise TypeError("Both signal_spect_db and noise_spect_db must be numpy arrays.")

        # Ensure both spectrograms have the same number of frequency bands
        if signal_spect_db.shape[0] != noise_spect_db.shape[0]:
            raise ValueError(
                f"Spectrograms must have the same number of frequency bands. "
                f"Got {signal_spect_db.shape[0]} and {noise_spect_db.shape[0]} bands."
            )

        # Apply frequency filtering
        signal_filtered, _ = self._filter_by_frequency_range(signal_spect_db, min_freq_hz, max_freq_hz)
        noise_filtered, _ = self._filter_by_frequency_range(noise_spect_db, min_freq_hz, max_freq_hz)

        # Estimate noise level using 95th percentile across time
        # To get a single value per frequency band that is representative of the noise level
        noise_db_per_band = np.quantile(noise_filtered, q=0.95, axis=1)[:, np.newaxis]    # Shape: (n_selected_mels, 1)

        # Compute SNR (in dB) over time
        snr_db = signal_filtered - noise_db_per_band    # Shape: (freq, time)

        #Collapse the frequency axis to get a single value per time point
        #To basically get where snr for each time point
        snr_db_per_time = np.max(snr_db, axis=0)                         # Shape: (time,)

        # Determine active time points where SNR exceeds threshold (where there is signal
        snr_time_mask = snr_db_per_time > snr_threshold
        
        try:
            true_indices = np.where(snr_time_mask)[0]
            whistle_start = np.percentile(true_indices, 10)/960
            whistle_end = np.percentile(true_indices, 90)/960

            data = true_indices

            bins = np.arange(0, 961, 96)

            
            hist, bin_edges = np.histogram(data, bins=bins)

            high_count_indices = np.where(hist > 15)[0]

            whistle_start = (high_count_indices[0])*0.1
            whistle_duration = (high_count_indices[-1]-high_count_indices[0]+1)*0.1

        
        except:
            whistle_start = 0
            whistle_duration = 0

        # Calculate the fraction of time points where SNR exceeds threshold
        active_time_fraction = np.mean(snr_time_mask)

        avg_snr_signal = np.mean(snr_db_per_time[snr_time_mask]) if np.any(snr_time_mask) else 0
        
        # Compute statistics
        snr_stats = {
            'avg_window': round(np.mean(snr_db_per_time), 2),
            'signal_active_time': round(active_time_fraction, 2),
            'signal_present': int(active_time_fraction > time_threshold),
            'avg_signal': round(avg_snr_signal, 2),
            'whistle_start': whistle_start,
            'whistle_duration': whistle_duration,
        }

        return snr_stats

SPECT_GENERATOR = SpectrogramGenerator(
        n_fft=2048,
        hop_length=200,
        n_mels=64,
        fmin=200,
        sample_rate=192000, 
    )

def get_audio_start_time(audio_file):
    """
    Extract and parse the start time from an audio filename.
    
    Expected format: hydrophone_id.YYMMDDHHMMSS.wav
    
    Args:
        audio_file (str): The audio filename
        
    Returns:
        datetime: The parsed start time
        
    Raises:
        ValueError: If the file doesn't end with .wav or doesn't contain a valid date
    """
    # Check if file ends with .wav
    if not audio_file.endswith('.wav'):
        raise ValueError(f"File '{audio_file}' is not a WAV file")
    
    # Split the filename by dots and try to extract the date part
    parts = audio_file.split(".")
    
    # Check if we have enough parts (should have at least 3: hydrophone_id.date.wav)
    if len(parts) < 3:
        raise ValueError(f"Filename '{audio_file}' does not contain a date component")
    
    date_str = parts[1]
    
    # Try to parse the date string
    try:
        audio_start_time = datetime.strptime(date_str, "%y%m%d%H%M%S")
        return audio_start_time
    except ValueError:
        raise ValueError(f"Could not parse date from '{date_str}' in filename '{audio_file}'")

def compute_initial_noise_spects(audio, sample_rate, call_model, spect_generator=SPECT_GENERATOR, hydrophone_sensitivity=None, call_model_window_s=1.5, compute_n_noise_spects=5, debug=False, max_steps=100):
    """
    Computes and returns a list of spectrograms containing no beluga calls based on the classificiation results of our call model.

    Only spectrograms classified as not containing a whistle or high-frequency call
    are stored as 'noise'.

    Parameters:
    - audio: np.ndarray, raw audio signal
    - sample_rate: int, sample rate of the audio
    - call_model: classifier model to detect calls
    - call_model_window_s: float, time window in seconds
    - compute_n_noise_spects: int, number of noise spectrograms to collect
    - debug: bool, if True, prints debug info
    - max_steps: int, maximum number of steps to compute before giving up
    Returns:
    - List of dB spectrograms (length: n_noise_spects)
    """

    if compute_n_noise_spects==0:
        return []

    window_samples = int(call_model_window_s * sample_rate)
    total_samples = len(audio)
    step = window_samples

    noise_spects = []
    current_sample_idx = 0
    i = 0

    while current_sample_idx + window_samples <= total_samples and len(noise_spects) < compute_n_noise_spects:
        snippet = audio[current_sample_idx:current_sample_idx+window_samples]

        power_spect = spect_generator.compute_mel_power_spect(snippet)
        dB_spect = spect_generator.power_to_db(power_spect, hydrophone_sensitivity=hydrophone_sensitivity)
        # dB_spect = SPECT_GENERATOR.power_to_db(power_spect)
        normalized_spect = spect_generator.min_max_normalization(dB_spect)
        resized_spect = SPECT_GENERATOR.resize_spect(normalized_spect, img_shape=(244, 244))

        res = classify_overlap(resized_spect, call_model)
        prediction, probs = res[0]

        # if debug:
        #     print(f"Window {i // step + 1}: Prediction = {prediction}")
        #     SPECT_GENERATOR.plot_spect(dB_spect, title=f"Model prediction : {prediction} => NOISE")


        if probs["Call_Detection"] < 0.1:

            # We remove the first and last bits of the spectrogram to avoid edge effects of spectrogram generation
            noise_spect_more_confident = dB_spect[:, 10:-10]
            noise_spects.append(noise_spect_more_confident)
            # noise_spects.append(dB_spect)
            # SPECT_GENERATOR.plot_spect(dB_spect, title=f"Model prediction : {prediction} => NOISE")
            # SPECT_GENERATOR.plot_spect(noise_spect_more_confident, title=f"Model prediction : {prediction} => NOISE")

        current_sample_idx += step
        i += 1

        if i > max_steps:
            if debug:
                print(f"WARNING max steps reached ({max_steps}) while computing initial noise spectrograms, found {len(noise_spects)} noise spectrograms")
            break


    if debug:
        
        # print(f"Found the noise spects in the first {i/step} computed spectrograms")
        if len(noise_spects) > 0:
            concatenated_noise_spect = np.concatenate(noise_spects, axis=1)
            SPECT_GENERATOR.plot_spect(concatenated_noise_spect, title=f"{compute_n_noise_spects} concatenated initial noise spectrograms")

    return noise_spects

def classify_overlap(
    spectrograms, 
    model, 
    label_columns=["ECHO", "HFPC", "CC", "Whistle"], 
    batch_size=32, 
    threshold=0.5
):
    """
    Classify spectrograms using a multi-label model in mini-batches.
    Handles both lists of spectrograms and a single spectrogram.

    Args:
        spectrograms (list or np.ndarray): List of spectrograms or a single spectrogram (H, W).
        model (torch.nn.Module): Trained multi-label classification model.
        label_columns (list of str): Names of the output labels.
        batch_size (int): Number of spectrograms to process per batch.
        threshold (float): Threshold for considering a label as present.

    Returns:
        list of tuples: Each tuple is (predicted_labels_dict, probabilities_dict)
    """
    all_results = []

    # Handle a single spectrogram input
    if isinstance(spectrograms, np.ndarray) and spectrograms.ndim == 2:
        spectrograms = [spectrograms]  

    device = model.device

    for i in range(0, len(spectrograms), batch_size):
        batch = spectrograms[i:i + batch_size]
        tensor_batch = torch.from_numpy(np.stack(batch)).float().unsqueeze(1)  # [B, 1, H, W]
        tensor_batch = tensor_batch.to(device)

        with torch.no_grad():
            logits = model(tensor_batch)
            probabilities = torch.sigmoid(logits)

        for probs in probabilities:
            probs = probs.cpu().numpy()
            prob_dict = {label: round(float(prob), 3) for label, prob in zip(label_columns, probs)}
            prob_dict["Call_Detection"] = round(float(max(probs)), 3)

            pred_dict = {
                label: (prob >= threshold) 
                for label, prob in prob_dict.items()
            }

            
            all_results.append((pred_dict, prob_dict))

    
    return all_results 

def calculate_window_db_snr(dB_spect, noise_profile_spect, whistles_freq_range, hf_calls_freq_range, spect_generator):
    db_window = {
        "w_range": spect_generator.compute_db_in_freq_range(dB_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=whistles_freq_range[1]),
        "hf_range": spect_generator.compute_db_in_freq_range(dB_spect, min_freq_hz=hf_calls_freq_range[0], max_freq_hz=hf_calls_freq_range[1]),
        # "full_range": spect_generator.compute_db_in_freq_range(dB_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=hf_calls_freq_range[1])
    }

    db_noise = {
        "w_range": spect_generator.compute_db_in_freq_range(noise_profile_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=whistles_freq_range[1]),
        "hf_range": spect_generator.compute_db_in_freq_range(noise_profile_spect, min_freq_hz=hf_calls_freq_range[0], max_freq_hz=hf_calls_freq_range[1]),
        # "full_range": spect_generator.compute_db_in_freq_range(noise_profile_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=hf_calls_freq_range[1])
    }
    snr = {
        "w_range": spect_generator.compute_snr_through_time(dB_spect, noise_profile_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=whistles_freq_range[1]),
        "hf_range": spect_generator.compute_snr_through_time(dB_spect, noise_profile_spect, min_freq_hz=hf_calls_freq_range[0], max_freq_hz=hf_calls_freq_range[1]),
        # "full_range": spect_generator.compute_snr_through_time(dB_spect, noise_profile_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=hf_calls_freq_range[1])
    }
    # peak_snr = {
    #     "w_range": spect_generator.compute_peak_snr_with_time_compressed(dB_spect, noise_profile_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=whistles_freq_range[1]),
    #     "hf_range": spect_generator.compute_peak_snr_with_time_compressed(dB_spect, noise_profile_spect, min_freq_hz=hf_calls_freq_range[0], max_freq_hz=hf_calls_freq_range[1]),
    #     "full_range": spect_generator.compute_peak_snr_with_time_compressed(dB_spect, noise_profile_spect, min_freq_hz=whistles_freq_range[0], max_freq_hz=hf_calls_freq_range[1])
    # }

    return db_window, db_noise, snr

def run_pipeline_overlaps_long_spects(
    audio,
    audio_start_time,
    sample_rate,
    model,
    debug=False,
    hydrophone_sensitivity=None,
    call_model_window_s=1,
    seconds_to_process=None,
    batch_size=32,
    debug_at_s=None,
    n_noise_spects=5,
    whistles_freq_range=(500, 25000),
    hf_calls_freq_range=(25000, 190000),
    process_index=0,
    spect_generator=SPECT_GENERATOR,
    skip_first_n_seconds=0
):
    """
    Process an audio file using batch inference instead of single snippets.
    Uses an optimized approach that generates spectrograms for longer segments at once.
    """
    print("Running pipeline no ROIs (Batched with optimized spectrogram generation)")
    
    # Duration calculations
    audio_duration = len(audio) / sample_rate
    end_time = audio_start_time + timedelta(seconds=audio_duration)

    if debug:
        print(f"Audio is {audio_duration:.2f} seconds long")

    if seconds_to_process is not None and seconds_to_process < audio_duration:
        end_time = audio_start_time + timedelta(seconds=seconds_to_process)
        print(f"Processing first {seconds_to_process} seconds.")

    step_duration = call_model_window_s
    current_time = audio_start_time + timedelta(seconds=skip_first_n_seconds)

    # current_time = audio_start_time + timedelta(seconds=4)

    results = []

    # Timing variables
    spects_generation_time = 0
    spect_initial_gen_time = 0
    spect_db_time = 0
    spect_normalization_time = 0
    spect_resizing_time = 0
    inference_time = 0
    noise_calculation_time = 0

    # Noise profile setup
    noise_spects = []
    noise_profile_spect = None

    if debug:
        print(f"Computing {n_noise_spects} spectrograms with only noise, to then be used to create a reliable noise profile")
      
    noise_spects = compute_initial_noise_spects(
        audio, sample_rate, model, hydrophone_sensitivity=hydrophone_sensitivity, 
        call_model_window_s=call_model_window_s, compute_n_noise_spects=5, 
        debug=False, max_steps=500)

    # noise_spects = []
    
    if len(noise_spects) == 0:
        print("WARNING no noise spect found")
    else:
        noise_profile_spect = np.concatenate(noise_spects, axis=1)
     
    total_steps = int(((end_time - audio_start_time).total_seconds()) // step_duration) + 1

    if debug_at_s is not None:
        debug = False

    # Calculate samples per window and time bins per window for spectrogram slicing
    
    time_bins_per_window = int(call_model_window_s * sample_rate / spect_generator.hop_length)
    
    with tqdm(total=total_steps, desc=f"Process {process_index}", position=process_index, leave=True) as pbar:
        # Process audio in chunks of batch_size * call_model_window_s
        while current_time <= end_time:
            elapsed_time_seconds = (current_time - audio_start_time).total_seconds()
            if (debug_at_s is not None) and elapsed_time_seconds > debug_at_s:
                debug = True
            
            # Calculate how many full windows we can process in this batch
            remaining_seconds = (end_time - current_time).total_seconds()
            windows_in_batch = min(batch_size, int(remaining_seconds / call_model_window_s) + 1)
            
            if windows_in_batch <= 0:
                break
                
            # Calculate audio indices for the entire batch segment
            batch_start_index = int(elapsed_time_seconds * sample_rate)
            batch_duration = windows_in_batch * call_model_window_s
            batch_end_index = min(batch_start_index + int(batch_duration * sample_rate), len(audio))
            
            # Skip if we're at the end of the audio
            if batch_start_index >= len(audio):
                break
                
            # Extract the batch audio segment
            batch_audio = audio[batch_start_index:batch_end_index]
            
            # Generate one long spectrogram for the entire batch
            time_calc_start = time.time()
            long_power_spect = spect_generator.compute_mel_power_spect(batch_audio)
            spect_initial_gen_time += time.time() - time_calc_start
            
            time_calc_start = time.time()
            long_db_spect = spect_generator.power_to_db(long_power_spect, hydrophone_sensitivity=hydrophone_sensitivity)
            # long_db_spect = spect_generator.power_to_db(long_power_spect)
            spect_db_time += time.time() - time_calc_start

            # spect_generator.plot_spect(long_db_spect, title="Long spectrogram", vmax=160, vmin=70)
            # spect_generator.plot_spect(long_db_spect, title="Long spectrogram", vmax=80, vmin=0)
            # spect_generator.plot_spect(long_db_spect, title="Long spectrogram", vmax=160, vmin=70)
            
            # Calculate actual number of windows we can extract from the spectrogram
            actual_windows = min(windows_in_batch, long_db_spect.shape[1] // time_bins_per_window)
            
            # Prepare batch data
            batch_db_spects = []
            batch_model_spects = []
            batch_start_times = []


            used_median_noise_profile = False
            if len(noise_spects) == 0:
                if debug:
                    print("Computing artificial noise profile")
                noise_profile_spect = np.median(long_db_spect, axis=1)[:, np.newaxis]
                used_median_noise_profile = True
            

            # Cut the long spectrogram into windows
            for i in range(actual_windows): 
                start_bin = i * time_bins_per_window
                end_bin = (i + 1) * time_bins_per_window
                
                # Skip if we don't have enough bins left
                if end_bin > long_db_spect.shape[1]:
                    break
                    
                window_db_spect = long_db_spect[:, start_bin:end_bin]
                batch_db_spects.append(window_db_spect)
                
                # Normalize and resize for model input
                time_calc_start = time.time()
                model_normalized_spect = spect_generator.min_max_normalization(window_db_spect)
                spect_normalization_time += time.time() - time_calc_start

                # batch_model_spects.append(model_normalized_spect)
                
                time_calc_start = time.time()
                model_resized_spect = spect_generator.resize_spect(model_normalized_spect, img_shape=(244, 244))
                spect_resizing_time += time.time() - time_calc_start
                
                batch_model_spects.append(model_resized_spect)
                
                # Calculate timestamp for this window
                window_start_time = current_time + timedelta(seconds=i * call_model_window_s)
                batch_start_times.append(window_start_time)
            
            # Run inference on the batch
            if batch_model_spects:

                time_calc_start = time.time()
                # Classify the batch
                batch_predictions = classify_overlap(batch_model_spects, model, batch_size=batch_size)
                inference_time += time.time() - time_calc_start

                # Process predictions
          
                for (pred, probs), dB_spect, start_time_single in zip(batch_predictions, batch_db_spects, batch_start_times):
                    time_calc_start = time.time()


                    db_window, db_noise, snr = calculate_window_db_snr(dB_spect, noise_profile_spect, whistles_freq_range, hf_calls_freq_range, spect_generator)
                
                    noise_calculation_time += time.time() - time_calc_start

                    if debug:
                        print(f"Seconds since start: {(start_time_single - audio_start_time).total_seconds()}")
                        print(pred)
                        print(probs)
                        spect_generator.plot_spect(dB_spect, title=f"ECHO: {pred['ECHO']}, HFPC: {pred['HFPC']}, CC: {pred['CC']}, Whistle: {pred['Whistle']}, Call: {pred['Call_Detection']}", figsize=(5, 2.5), vmax=160, vmin=70)
                        
                    if probs["Call_Detection"] < 0.1 :  
                        # We remove the first and last bits of the spectrogram to avoid edge effects of spectrogram generation
                        new_noise_spect = dB_spect[:, 10:-10]

                        if len(noise_spects) == 0:
                            noise_spects.append(new_noise_spect)
                        else:
                            #Let's check if the new noise spect intensity is lower than the previous ones
                            if np.mean(new_noise_spect) < np.mean(noise_profile_spect):
                                #It is lower so let's rewrite the noise spects to make sure we keep the lowest most recent noise profile
                                if debug:
                                    print("Replacing noise profile because new one is lower")
                                noise_spects = [new_noise_spect]
                            else:
                                #It is higher we want to keep to simply append it to the list, in case it's a false negative and we actually have a bit of call in the snippet
                                if len(noise_spects) >= n_noise_spects:
                                    noise_spects.pop(0)
                                noise_spects.append(new_noise_spect)
                    

                        noise_profile_spect = np.concatenate(noise_spects, axis=1)
                        if debug:
                            print("Updating noise profile")
                            print("This snippet has no calls => UPDATE NOISE PROFILE")
                            SPECT_GENERATOR.plot_spect(noise_profile_spect, title="New concatenated noise spectrograms", vmax=180, vmin=70)
                        

                    noise_calculation_method = "precise_with_model" if not used_median_noise_profile else "default_median_value"

                    result = {
                        "Timestamp": start_time_single,
                        "seconds_since_file_start": (start_time_single - audio_start_time).total_seconds(),
                        "noise_calc_method": noise_calculation_method,
                        "Call_Detection": probs.get("Call_Detection", 0.0),  
                    }
                    # Add call presence and probabilities
                    for label in ["ECHO", "HFPC", "CC", "Whistle"]:
                        result[f"{label}"] = pred.get(label, False)
                        result[f"{label}_prob"] = probs.get(label, 0.0)
                    

                    #mb55 Here if I want to modify the results to the detection it's here 


                    # if  True : # i == 0:
                    #     result['HFPC'] = True
                    #     result['HFPC_prob']= 1.0
                    #     result['Call_Detection'] = 1.0
                    #     for label in ["ECHO", "CC", "Whistle"]:
                    #         result[f"{label}"] = False
                    #         result[f"{label}_prob"] = 0.0
                    # # else: 
                    #     result['HFPC'] = False
                    #     result['HFPC_prob']= 0.0
                    
                    if True : # i%5 == 0:
                        result['Whistle'] = True
                        result['Whistle_prob']= 1.0
                        result['Call_Detection'] = 1.0

                        for label in ["HFPC", "CC", "ECHO"]:
                            result[f"{label}"] = False
                            result[f"{label}_prob"] = 0.0
                    
                    # else: 
                    #     result['Whistle'] = False
                    #     result['Whistle_prob']= 0.0
                     

                        
                    #end of modification mb55
                    
                    # Add dB window and noise values
                    for band in ["w_range", "hf_range"]:
                        for stat in db_window[band].keys():
                            result[f"db_window_{band}_{stat}"] = round(db_window[band][stat], 1)
                            result[f"db_noise_{band}_{stat}"] = round(db_noise[band][stat], 1)
                        
                        for stat in snr[band].keys():
                            result[f"snr_{band}_{stat}"] = round(snr[band][stat], 2)

                    
                    # Add the result to the results list
                    results.append(result)
            
            # Move forward by the number of windows we actually processed
            current_time += timedelta(seconds=batch_duration)
            pbar.update(actual_windows)

    # columns = ["Call", "Type", "Duration", "Timestamp", "seconds_since_file_start", "min_freq_hz", "max_freq_hz", "noise_calculation_method", "db_noise", "SNR_window", "SNR_peak", "probabilities", "detection_confidence"]
    # results_df = pd.DataFrame(results, columns=columns)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Timestamp")
    
    spects_generation_time = spect_initial_gen_time + spect_db_time + spect_normalization_time + spect_resizing_time
    print("===============")
    print(f"Spects generation time: {spects_generation_time:.2f} seconds")
    print(f"  Initial generation: {spect_initial_gen_time:.2f} seconds")
    print(f"  dB conversion: {spect_db_time:.2f} seconds") 
    print(f"  Normalization: {spect_normalization_time:.2f} seconds")
    print(f"  Resizing: {spect_resizing_time:.2f} seconds")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Noise calculation time: {noise_calculation_time:.2f} seconds")
    print("===============")
    return results_df