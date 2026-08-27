from time_frequency_mask.config import Parameters
from time_frequency_mask.plotter import plot_spectrogram_4D
from time_frequency_mask.data_generation.core.sampling import sample_duration, sample_whistles, sample_impulsive_noise, sample_shifts, sample_snrs
from time_frequency_mask.data_generation.models.audio_sample import LabeledAudioSample, TetrahedraAudioSample
from time_frequency_mask.data_generation.io.data_parser import  retrieve_wav_and_masks_paths

class Generator():
    def __init__(self, parameters : Parameters, showcase):
        self.parameters : Parameters = parameters
        self.num_samples : int = parameters.generation.num_audio_samples
        self.showcase : bool = showcase
        self.wav_and_masks_paths = retrieve_wav_and_masks_paths(parameters.generation.whistle_bank_path)

    def sample_data(self):
        duration = sample_duration(self.parameters.audio)
        whistles = sample_whistles(self.wav_and_masks_paths, self.parameters, duration)
        shifts = sample_shifts(self.parameters)
        snrs = sample_snrs(self.parameters)
        return duration, whistles, shifts, snrs

    def generate_sample(self):
        duration, whistles, shifts, snrs = self.sample_data()

        labeled_audio_sample = LabeledAudioSample.from_empty_wav(self.parameters, duration)

        for whistle in whistles:
            labeled_audio_sample += whistle

        if self.parameters.noise.enable_impulsive_noise:
            for impulsive_noise in sample_impulsive_noise():
                labeled_audio_sample += impulsive_noise

        tetrahedra_audio_sample = TetrahedraAudioSample.from_single_labeled_audio_sample(labeled_audio_sample, self.parameters.array.num_mics)

        tetrahedra_audio_sample.set_tdoas(self.parameters, shifts)

        tetrahedra_audio_sample.set_gaussian_noise(snrs, self.parameters)

        return tetrahedra_audio_sample

    def save_sample(self, tetrahedra_audio_sample : TetrahedraAudioSample, sample_idx : int, parameters : Parameters):
        stem = f"sample_{self.parameters.generation.wav_count + sample_idx}"
        tetrahedra_audio_sample.save(str(parameters.generation.output_dir), stem, parameters)

    def run(self):
        for sample_idx in range(self.parameters.generation.num_audio_samples):
            tetrahedra_audio_sample = self.generate_sample()
            if self.showcase:

                plot_spectrogram_4D(
                    tetrahedra_audio_sample.shifted_waveforms,
                    tetrahedra_audio_sample.sampling_rate,
                    self.parameters,
                    mask=tetrahedra_audio_sample.shifted_masks[0].data,
                    is_db=False,
                )
            else:
                self.save_sample(tetrahedra_audio_sample, sample_idx, self.parameters)