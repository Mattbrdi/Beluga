from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beamforming.simulation.generate_audio import generate_audio_from_source_and_tetrahedra  
from BSS.Utils.signal_class import Signal, MultiSignal
from time_frequency_mask.data_generation.models.mask import AudioMask
from time_frequency_mask.masknet.run_inference import get_mask_from_array
from time_frequency_mask.plotter import plot_spectrogram_4D

from beamforming.classes import * 
from beamforming.mathematics.wideband_beamforming import wideband_cssm_music_optimized_doa, wideband_cssm_mvdr_optimized_doa
from beamforming.simulation.visualization import visualize_scene_3D, plot_results
from beamforming.mathematics.masked_beamforming import masked_music_doa, masked_mvdr_doa
is_tf_mask = True

mask = None
std_noise = 5
rng = np.random.default_rng(seed=42)  


source_pos = 100*np.array([1, 0, -np.sqrt(2 / 3)])

regular_tetrahedra = TetrahedralArray([
        [0, 0, 0],
        [1, 0, 0],
        [1 / 2, np.sqrt(3) / 2, 0],
        [1 / 2, np.sqrt(3) / 6, np.sqrt(2 / 3)],
    ])

freqs = [16000, 15000, 12000, 8000]
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

if is_tf_mask:
    inference_start = time()
    mask = AudioMask(get_mask_from_array(audio_array)).data
    inference_duration = time() - inference_start
    plot_spectrogram_4D(audio_array, SAMPLING_RATE, is_db=False, mask = mask)
# visualize_scene_3D(tetrahedra, source)
print("Beamforming Doa processing")
start = time()
u_narrow_masked_mvdr = masked_mvdr_doa(tetrahedra, audio_array)
t1 = time()
u_cssm_music = wideband_cssm_music_optimized_doa(tetrahedra, audio_array, center_freq, tf_mask=mask)
t2 = time()
u_narrow_masked_music = masked_music_doa(tetrahedra, audio_array)
t3 = time()
u_cssm_mvdr = wideband_cssm_mvdr_optimized_doa(tetrahedra, audio_array, center_freq, tf_mask=mask)

end = time()
# u = cssm_mvdr_optim_doa(tetrahedra, audio_array, center_freq)

print("========================================")
print(f"MASKED MVDR Error: {100*np.linalg.norm((u_narrow_masked_mvdr - u_true) / np.linalg.norm(u_cssm_mvdr))} and took {t1 - start + inference_duration}s")
print(f"CSSM MUSIC Error: {100*np.linalg.norm((u_cssm_music - u_true) / np.linalg.norm(u_cssm_music))} and took {t2 - t1 + inference_duration}s")
print(f"MASKED MUSIC Error: {100*np.linalg.norm((u_narrow_masked_music - u_true) / np.linalg.norm(u_narrow_masked_music))} and took {t3 - t2}s")
print(f"CSSM MVDR Error: {100*np.linalg.norm((u_cssm_mvdr - u_true) / np.linalg.norm(u_narrow_masked_mvdr))} and took {end - t3}s")

print("Visualize result")

