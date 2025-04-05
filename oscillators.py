import numpy as np


class Oscillator:
    # Sine osc bank with freq, amp and phase interpolation

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.phase = 0.0
        self.freq = 0.0
        self.amp = 0.0

    def process(self, target_freq, target_amp, nSamples):

        output = np.zeros(nSamples)

        freq_trajectory = np.linspace(
            self.freq, target_freq, nSamples
        )  # interpolate from set freq to target freq (i.e. next freq)
        amp_trajectory = np.linspace(
            self.amp, target_amp, nSamples
        )  # same as interpolation for freq

        for i in range(nSamples):

            phase_inc = (
                2.0 * np.pi * freq_trajectory[i] / self.sample_rate
            )  # dividing by fs gives the actual relative amount by which the phase should be incremented by

            self.phase += phase_inc

            while self.phase >= 2.0 * np.pi:
                self.phase -= 2.0 * np.pi

            output[i] = amp_trajectory[i] * np.sin(
                self.phase
            )  # this the sin wave created at every incremental phase

        self.freq = target_freq
        self.amp = target_amp

        return output


class AdditiveResynthesizer:

    def __init__(self, sample_rate, hop_size):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.oscillators = []

    def load_analysis_data(self, freq_file, amp_file):

        self.frequencies = np.loadtxt(freq_file)
        self.amplitudes = np.loadtxt(amp_file)

        self.num_partials = self.frequencies.shape[
            1
        ]  # number of actual frequencies or so called "partials"
        self.num_frames = self.frequencies.shape[
            0
        ]  # total number of frames from our windowing method to find f0

        self.oscillators = [
            Oscillator(self.sample_rate) for _ in range(self.num_partials)
        ]

    def synthesize(self):

        output_length = self.hop_size * (self.num_frames - 1) + self.hop_size
        output_audio = np.zeros(output_length)

        for frame in range(
            self.num_frames - 1
        ):  # -1 because interpolating between frames
            frame_pos = frame * self.hop_size

            for p in range(self.num_partials):

                curr_freq = self.frequencies[
                    frame, p
                ]  # this sets the current index in freq array
                next_freq = self.frequencies[
                    frame + 1, p
                ]  # this sets the next index in freq array
                curr_amp = self.amplitudes[
                    frame, p
                ]  # this sets the current index in amp array
                next_amp = self.amplitudes[
                    frame + 1, p
                ]  # this sets the next index in amp array

                if curr_freq == 0 or next_freq == 0:
                    continue

                samples = self.oscillators[p].process(
                    next_freq, next_amp, self.hop_size
                )  # process function to set the sine wave for ith partial

                output_audio[frame_pos : frame_pos + self.hop_size] += samples

        max_amp = np.max(np.abs(output_audio))
        if max_amp > 0:
            output_audio = output_audio / max_amp

        return output_audio
