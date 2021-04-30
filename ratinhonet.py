import numpy as np
import sounddevice as sd
import soundfile as sf

CHANNELS = 1
LOG = True
COOLDOWN = 500
UPDATE_WEIGHT = 0.03
CUT_OFF = 10

countdown = COOLDOWN
old_mean = 0
ratinho_sound, ratinho_sample_rate = sf.read("audio/fart.wav")


def callback(amplitudes, frames, time, status):
    global countdown, old_mean

    volumes = np.abs(amplitudes)
    current_mean = np.mean(volumes)
    mean_ratio = current_mean / old_mean

    if LOG:
        print(f"countdown: {countdown}, mean ratio: {mean_ratio:.2f}")

    if countdown == 0 and mean_ratio > CUT_OFF:
        countdown = COOLDOWN
        sd.play(ratinho_sound, ratinho_sample_rate)

    countdown -= 1
    countdown = 0 if countdown < 0 else countdown
    old_mean = (1 - UPDATE_WEIGHT) * old_mean + UPDATE_WEIGHT * current_mean


with sd.InputStream(channels=CHANNELS, callback=callback):
    input()
