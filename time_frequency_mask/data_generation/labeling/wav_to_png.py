import numpy as np

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from time_frequency_mask.data_generation.io.data_parser import read_wav_file, save_wav_file
from time_frequency_mask.stft import scipy_db_spectrogram, frequency_band, scipy_spectrogram, scipy_stft

PATH = r"C:\Users\amine\Desktop\Canada\Beluga\time_frequency_mask\data_generation\data\input\whistle2"

import matplotlib.pyplot as plt
for waveform_path in Path(PATH).glob("*.wav"):
    waveform_name = str(waveform_path).split('.')[0]
    waveform, sampling_rate = read_wav_file(str(waveform_path), num_canals=1)
    # freqs, times, D = scipy_db_spectrogram(waveform, sampling_rate, n_fft=4096, hop_length=2048)
    freqs, times, D = scipy_spectrogram(waveform, sampling_rate)
    # vmin = -120
    # vmax = -20
    # colorbar_format = '%+2.0f dB'

    freqs, D = frequency_band(freqs, D, 500, 20000)
    extent = [
        times[0],
        times[-1] if len(times) > 1 else 0,
        freqs[0],
        freqs[-1],
    ]
    print(times.shape, freqs.shape)
    print(np.shape(extent))
    D = D - np.min(D)

    D = D / np.percentile(np.abs(D),99)

    D = np.clip(D, 0,1)

    D = np.array(plt.cm.magma(D))
    
    D = np.round(D*255)
    from PIL import Image

    # Example: Create a 100x100 RGB random image array
    # 1. Convert the data type to uint8
    D = D.astype(np.uint8)

    # 2. Convert the NumPy array to a Pillow Image object
    img = Image.fromarray(D)
    output_folder = Path(PATH)
    output_path = output_folder / "png" / f"{waveform_path.stem}.png"    
    img.save(output_path)

    # fig, ax = plt.subplots(1, 1, figsize=(30, 5))
    # mappable = ax.imshow(
    #     D,
    #     origin='lower',
    #     aspect='auto',
    #     extent=extent,
    #     cmap="magma",
    #     vmin=vmin,
    #     vmax=vmax,
    # )

    # ax.set_title('Spectrogramme SciPy du Canal')
    # # ax.set_ylim([500,20000])
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Frequency (Hz)')
    # plt.colorbar(mappable, ax=ax, format=colorbar_format)
    # plt.tight_layout()
    # plt.show()
    
    # output_folder = Path(PATH)
    # output_path = output_folder / "png" / f"{waveform_path.stem}.png"    
    # plt.savefig(output_path)
    # plt.close()

    # import cv2
    # import matplotlib.pyplot as plt

    # # 1. Load the image in grayscale
    # gray_image = cv2.imread('your_image.png', cv2.IMREAD_GRAYSCALE)

    # # 2. Apply the Magma colormap
    # magma_image = cv2.applyColorMap(D, cv2.COLORMAP_MAGMA)

    # # 3. Save or display the result
    # cv2.imwrite('magma_output.png', magma_image)
