from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from beamforming.simulation.generate_audio import generate_audio_from_source_and_tetrahedra  
from BSS.Utils.signal_class import Signal
from beamforming.classes import * 
from beamforming.mathematics.beamformer import mvdr, mvdr_doa, music, delay_and_sum, music_doa, delay_and_sum_doa
from beamforming.simulation.visualization import visualize_scene_3D, plot_results

source_pos = 100*np.array([1, 0, np.sqrt(2 / 3)])

regular_tetrahedra = TetrahedralArray([
        [0, 0, 0],
        [1, 0, 0],
        [1 / 2, np.sqrt(3) / 2, 0],
        [1 / 2, np.sqrt(3) / 6, np.sqrt(2 / 3)],
    ])

print("Generating Data")
s1 = Signal.generate_multi_freq_signal(
        1, 384000, start_time=0.15, end_time=0.7, 
        frequencies=[2000], window_type='hann'
    )

tetrahedra = TetrahedralArray.from_length_tetrahedra_centroid(0.3)

center = tetrahedra.center
u_true = (source_pos - center) / np.linalg.norm(source_pos - center)
print(f"direction is : {u_true}")

source = Source(s1.data, source_pos)
audio_array = generate_audio_from_source_and_tetrahedra(tetrahedra, source).data

# visualize_scene_3D(tetrahedra, source)

print("Beamforming Doa processing")
# power_DB, Theta, Phi = mvdr(5200, tetrahedra, audio_array)
# power_DB, Theta, Phi = delay_and_sum(300, tetrahedra, audio_array)
# power_DB, Theta, Phi = music(5030, tetrahedra, audio_array, num_expected_signals=1)


# plot_results(power_DB, Theta, Phi)

# u = mvdr_doa(2000, tetrahedra, audio_array)
# u = delay_and_sum_doa(2000, tetrahedra, audio_array)
u = music_doa(2000, tetrahedra, audio_array, num_expected_signals=1)
print(u)
print("Visualize result")

