import numpy as np

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beamforming.config import * 
from BSS.Utils.signal_class import Signal
from beamforming.geometry import Source, TetrahedralArray
from beamforming.workflows.wideband.srp_phat import SRPPHAT
from beamforming.workflows.wideband.tops import TOPS
from beamforming.workflows.wideband.cssm import CSSM
from beamforming.workflows.wideband.issm import ISSM
from beamforming.workflows.wideband.masked import MaskedBeamformer
from beamforming.beamformers.mvdr import MVDR
from beamforming.beamformers.music import MUSIC
from beamforming.grid import SearchGrid
from beamforming.results import PseudoSpectrumResult
from beamforming.simulation.generate_audio_array import generate_audio_from_source_and_tetrahedra
from beamforming.signal.stft import STFTConfig, compute_band_stft

from time_frequency_mask.plotter import plot_spectrogram_4D
from time_frequency_mask.data_generation.models.mask import AudioMask
from time_frequency_mask.masknet.run_inference import get_mask_from_array
from time_frequency_mask.tdoa_estimation.blob import Blob, blob_filtering_heuristic, output_blobs_from_mask, output_mask_from_blobs
is_tf_mask = False

mask = None
std_noise = 1
rng = np.random.default_rng(seed=42)  


source_pos = 100*np.array([1, 0, -np.sqrt(2 / 3)])

regular_tetrahedra = TetrahedralArray([
        [0, 0, 0],
        [1, 0, 0],
        [1 / 2, np.sqrt(3) / 2, 0],
        [1 / 2, np.sqrt(3) / 6, np.sqrt(2 / 3)],
    ])

# freqs = [16000, 15000, 12000, 8000, 18000]
freqs = [1000, 1200, 1400, 1600]
center_freq = min(np.mean(freqs), 2000)

print("Generating Data")
s1 = Signal.generate_multi_freq_signal(
        1, 384000, start_time=0.15, end_time=0.4, 
        frequencies=freqs, window_type='hann'
    )

tetrahedra = TetrahedralArray.from_length_tetrahedra_centroid(0.3)

center = tetrahedra.center
u_true = (source_pos - center) / np.linalg.norm(source_pos - center)
print(f"direction is : {u_true}")

source = Source(s1.data, source_pos)
audio_array = generate_audio_from_source_and_tetrahedra(tetrahedra, source).data
audio_array += rng.normal(loc=0, scale=std_noise, size = audio_array.shape)

from time import time
stft_config = STFTConfig()
freqs, times, Zxx = compute_band_stft(audio_array, stft_config, SAMPLING_RATE)
freqs_tops, Zxx_tops = freqs, Zxx
if is_tf_mask:
    inference_start = time()
    mask = AudioMask(get_mask_from_array(audio_array)).data
    blobs = output_blobs_from_mask(AudioMask(mask))
    blobs = blob_filtering_heuristic(blobs)
    mask = output_mask_from_blobs(blobs).data
    # plot_spectrogram_4D(audio_array, SAMPLING_RATE, is_db=False, mask = mask)

    weights = mask != 0
    Zxx = Zxx * weights[:, None, :]
    active = np.any(weights > 0, axis=1)
    Zxx_tops = Zxx[active]
    freqs_tops = freqs[active]
    inference_duration = time() - inference_start

freq_idx = np.argmin(np.abs(freqs_tops - center_freq))
grid = SearchGrid.full_sphere(200, 200)
music = MUSIC(1)
mvdr = MVDR()

# visualize_scene_3D(tetrahedra, source)
print("Beamforming Doa processing")
if is_tf_mask:
    start = time()
    u_narrow_masked_mvdr = MaskedBeamformer(tetrahedra, mvdr).compute(audio_array, grid, blobs).doa.vector
    t1 = time()
    u_narrow_masked_music = MaskedBeamformer(tetrahedra, music).compute(audio_array, grid, blobs).doa.vector
t2 = time()
u_issm_mvdr = ISSM(tetrahedra, mvdr, stft_config).compute_from_stft(Zxx_tops, freqs_tops, grid).doa
u_cssm_music = CSSM(tetrahedra, music, stft_config, center_freq, u_issm_mvdr).compute_from_stft(Zxx, freqs, grid).doa.vector
t3 = time()
u_cssm_mvdr = CSSM(tetrahedra, music, stft_config, center_freq, u_issm_mvdr).compute_from_stft(Zxx, freqs, grid).doa.vector
t4 = time()
u_tops = TOPS(tetrahedra, stft_config, 1, freq_idx).compute_from_stft(Zxx_tops, freqs_tops, grid).doa.vector
t5 = time()
if is_tf_mask:
    u_srp_phat = SRPPHAT(tetrahedra, stft_config).compute_weighted_from_stft(Zxx, freqs, grid, weights).doa.vector
else:
    u_srp_phat = SRPPHAT(tetrahedra, stft_config).compute_from_stft(Zxx, freqs, grid).doa.vector
end = time()
if not is_tf_mask:
    inference_duration = 0
print("========================================")
if is_tf_mask:
    print(f"MASKED MVDR Error: {100*np.linalg.norm((u_narrow_masked_mvdr - u_true) / np.linalg.norm(u_narrow_masked_mvdr))} and took {t1 - start + inference_duration}s")
    print(f"MASKED MUSIC Error: {100*np.linalg.norm((u_narrow_masked_music - u_true) / np.linalg.norm(u_narrow_masked_music))} and took {t2 - t1 + inference_duration}s")
print(f"CSSM MUSIC Error: {100*np.linalg.norm((u_cssm_music - u_true) / np.linalg.norm(u_cssm_music))} and took {t3 - t2 + inference_duration}s")
print(f"CSSM MVDR Error: {100*np.linalg.norm((u_cssm_mvdr - u_true) / np.linalg.norm(u_cssm_mvdr))} and took {t4 - t3 + inference_duration}s")
print(f"TOPS Error: {100*np.linalg.norm((u_tops - u_true) / np.linalg.norm(u_tops))} and took {t5 - t4 + inference_duration}s")
print(f"SRP-PHAT Error: {100*np.linalg.norm((u_srp_phat - u_true) / np.linalg.norm(u_srp_phat))} and took {end - t5 + inference_duration}s")


print("Visualize result")

