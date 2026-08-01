from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beamforming.simulation.generate_audio import generate_audio_from_source_and_tetrahedra  
from BSS.Utils.signal_class import Signal
from beamforming.classes import * 
from beamforming.mathematics.wideband_beamforming import wideband_cssm_music_optimized_doa, wideband_cssm_mvdr_optimized_doa
from beamforming.simulation.visualization import visualize_scene_3D, plot_results

source_pos = 100*np.array([1, 0, -np.sqrt(2 / 3)])

regular_tetrahedra = TetrahedralArray([
        [0, 0, 0],
        [1, 0, 0],
        [1 / 2, np.sqrt(3) / 2, 0],
        [1 / 2, np.sqrt(3) / 6, np.sqrt(2 / 3)],
    ])

freqs = [5000, 2000, 1100]
center_freq = np.mean(freqs)

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

# visualize_scene_3D(tetrahedra, source)

print("Beamforming Doa processing")
from time import time
start = time()
u = wideband_cssm_mvdr_optimized_doa(tetrahedra, audio_array, center_freq)
u = wideband_cssm_music_optimized_doa(tetrahedra, audio_array, center_freq)
# u = cssm_mvdr_optim_doa(tetrahedra, audio_array, center_freq)
print(f"beamforming took {time() - start}s")
print(u)
print("Visualize result")

