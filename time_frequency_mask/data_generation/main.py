import numpy as np 
import numpy.random as rd
from numpy.typing import NDArray

import argparse

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from time_frequency_mask.configuration import SAMPLING_RATE, DEBUG_LEVEL, OUTPUT_PATH, COUNT
from time_frequency_mask.plotter import plot_mask, plot_spectrogram_1D, plot_spectrogram_4D, plot_waveform_1D, plot_waveform_4D
from time_frequency_mask.data_generation.core.preprocess import preprocess
from time_frequency_mask.data_generation.core.sampling import sample, sample_impulsive_noise
from time_frequency_mask.data_generation.models.audio_sample import LabeledAudioSample, TetrahedraAudioSample

is_save = True
is_plot = False

# rd.default_rng(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic beluga mask generator")
    parser.add_argument("--output-path", help="Define output-path location", required=True, type=str)
    parser.add_argument("--num-samples" , help="number of generated samples", required=True, type=int)
    parser.add_argument("--enable-impulsive-noise", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    num_samples = args.num_samples
    if num_samples > 180: 
        raise ValueError(f"cannot generate more than 3 minutes of data at once, provided {args.num_samples} seconds")

    # if DEBUG_LEVEL >= 1:
    #     print(f"gaussian_noise_array_std {gaussian_noise_array_std}")
    #     print(f"gaussian_noise_array_amplitude {gaussian_noise_array_amplitude}")

    list_of_snrs = np.arange(15, -9, -0.5)
    print(len(list_of_snrs))
    for num_snr, snr in enumerate(list_of_snrs):
        print(f"{num_snr} / {len(list_of_snrs)}")
        list_of_tetrahedra_audio_sample : list[TetrahedraAudioSample] = []
        wav_dir = OUTPUT_PATH / "wav"
        COUNT = sum(1 for p in wav_dir.iterdir()) if wav_dir.exists() else 0
        for sample_idx  in range(num_samples):
            is_debug_current_sample_detailed = DEBUG_LEVEL >= 2 and (DEBUG_LEVEL >= 3 or sample_idx % 100 == 0)
            is_debug_current_sample = DEBUG_LEVEL >= 1 and (DEBUG_LEVEL >= 2 or sample_idx % 100 == 0)
            is_print_info = sample_idx  % 50 == 0 or is_debug_current_sample or is_debug_current_sample_detailed
            
            if is_print_info:
                print(f"sample {sample_idx} out of {num_samples}")

            labeled_audio_sample = LabeledAudioSample.from_empty_wav()

            num_whistles, start_times, shifts, whistles = sample(is_augmentation = False)

            impulsive_noise_samples = sample_impulsive_noise()

            if is_debug_current_sample_detailed:
                print(f"num_whistle : {num_whistles}")
            
            for whistle in whistles:
                labeled_audio_sample += whistle # Each whistle has a start time so __add__ is valid  

            if args.enable_impulsive_noise:
                for impulsive_noise in impulsive_noise_samples:
                    labeled_audio_sample += impulsive_noise

                if is_debug_current_sample_detailed:
                    plot_waveform_1D(labeled_audio_sample.waveform, SAMPLING_RATE)
                    plot_spectrogram_1D(labeled_audio_sample.waveform, SAMPLING_RATE)    
            
            # print("plot")
            # plot_waveform_1D(labeled_audio_sample.waveform, SAMPLING_RATE)
            # plot_spectrogram_1D(labeled_audio_sample.waveform, SAMPLING_RATE) 
            # plot_mask(labeled_audio_sample.mask.data)  
            # print("end plot")
            tetrahedra_audio_sample = TetrahedraAudioSample.from_single_labeled_audio_sample(labeled_audio_sample)
            tetrahedra_audio_sample.set_tdoas(shifts)

            #Currently set manually
            # snrs = np.array([rd.random()*5+0.5, rd.random()*10+0.5, rd.random()*3+12, rd.random()*12+9])
            snr_val = snr
            snrs = np.array([(rd.random()-0.5)*2+snr_val, (rd.random()-0.5)*1+snr_val, (rd.random()-0.5)*2+snr_val, (rd.random()-0.5)*2+snr_val])
            snrs = rd.permutation(snrs)
            
            tetrahedra_audio_sample.set_gaussian_noise(snrs)

            # print("plot")
            # plot_waveform_4D(tetrahedra_audio_sample.shifted_waveforms, SAMPLING_RATE)
            # plot_spectrogram_4D(tetrahedra_audio_sample.shifted_waveforms, SAMPLING_RATE, is_db=True) 
            # plot_mask(tetrahedra_audio_sample.shifted_masks[0].data)  
            # print("end plot")

            # if is_debug_current_sample_detailed:
            #     plot_waveform_4D([labeled_audio_sample.waveform for labeled_audio_sample in labeled_audio_samples], SAMPLING_RATE)
            #     plot_spectrogram_4D([labeled_audio_sample.waveform for labeled_audio_sample in labeled_audio_samples], SAMPLING_RATE)

            # if args.enable_impulsive_noise:
            #     #TODO: Adapt this function and have a metric that helps see if a whistle is visible or no if overlapping with impulsive noise
            #     for channel_idx, labeled_audio_sample in enumerate(labeled_audio_samples):
            #         labeled_audio_sample += noise_generator.impulsive_noise_generato_per_channel(gaussian_noise_array_std[channel_idx], gaussian_noise_array_amplitude[channel_idx])
            #         # audio_array[channel_idx] = waveform

            #     if is_debug_current_sample_detailed:
            #         plot_waveform_4D(audio_array, SAMPLING_RATE)
            #         plot_spectrogram_4D(audio_array, SAMPLING_RATE)    

            #     labeled_audio_samples = noise_generator.impulsive_noise_generator_per_audio_array(labeled_audio_samples, gaussian_noise_array_std[channel_idx], gaussian_noise_array_amplitude[channel_idx])

            if is_debug_current_sample:
                plot_waveform_4D(tetrahedra_audio_sample.shifted_waveforms, SAMPLING_RATE)
                plot_spectrogram_4D(tetrahedra_audio_sample.shifted_waveforms, SAMPLING_RATE)

            list_of_tetrahedra_audio_sample.append(tetrahedra_audio_sample)

        # preprocess(list_of_tetrahedra_audio_sample)

        if is_plot:
            for tetrahedra_audio_sample in list_of_tetrahedra_audio_sample:
                print("plot")
                # plot_waveform_4D(tetrahedra_audio_sample.shifted_waveforms, SAMPLING_RATE)

                def mask_ijk(i, j, k):
                    return tetrahedra_audio_sample.shifted_masks[i].data & tetrahedra_audio_sample.shifted_masks[j].data & tetrahedra_audio_sample.shifted_masks[k].data
                mask123 = mask_ijk(0, 1, 2)
                mask124 = mask_ijk(0, 1, 3)
                mask134 = mask_ijk(0, 2, 3)
                mask234 = mask_ijk(1, 2, 3)
                
                # mask1 = tetrahedra_audio_sample.shifted_masks[0].data & tetrahedra_audio_sample.shifted_masks[1].data & tetrahedra_audio_sample.shifted_masks[2].data & tetrahedra_audio_sample.shifted_masks[3].data
                mask = mask123 | mask124 | mask134 | mask234
                plot_spectrogram_4D(tetrahedra_audio_sample.shifted_waveforms, SAMPLING_RATE, is_db=False, mask=mask) 
                # plot_mask(tetrahedra_audio_sample.shifted_masks[0].data)
                print("end plot")

        if is_save:
            for sample_idx, tetrahedra_audio_sample in enumerate(list_of_tetrahedra_audio_sample):
                stem = f"sample_{COUNT + sample_idx}"
                tetrahedra_audio_sample.save(str(OUTPUT_PATH), stem)

if __name__ == "__main__":
    main()
