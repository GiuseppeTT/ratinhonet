import random
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import typer

CALLBACKS_PER_SECOND = 100
CHANNELS = 1


app = typer.Typer()


@app.command()
def run(
    source: Optional[str] = typer.Argument(
        None,
        help="Squeaks source path. If none, use built-in",
    ),
    cooldown: float = typer.Option(
        5,
        help="Wait at least --cooldown seconds before squeaking again",
    ),
    cut_off: float = typer.Option(
        10,
        help="How much INsensible RatinhoNet should be to sudden volume changes",
    ),
    verbose: bool = typer.Option(False),
):
    """
    Run RatinhoNet with SOURCE squeaks.

    Squeaks should be short (< 5 seconds) WAV files. The funny, the better.
    """

    sample_rate = compute_sample_rate()
    block_size = compute_block_size(sample_rate)

    squeaks = read_squeaks(source)
    initial_countdown = compute_initial_countdown(cooldown)
    callback = Callback(squeaks, initial_countdown, cut_off, verbose)

    stream = sd.InputStream(
        samplerate=sample_rate,
        blocksize=block_size,
        channels=CHANNELS,
        callback=callback,
    )

    stream.start()
    input("Press ENTER to stop.\n")
    stream.stop()


def compute_sample_rate():
    device, _ = sd.default.device
    device_informations = sd.query_devices(device)
    sample_rate = device_informations["default_samplerate"]

    return sample_rate


def compute_block_size(sample_rate):
    return int(sample_rate / CALLBACKS_PER_SECOND)


def read_squeaks(source):
    if source is None:
        source_path = Path(__file__).parent / "squeak"
    else:
        source_path = Path(source)

    squeaks = [sf.read(file) for file in source_path.iterdir()]

    return squeaks


def compute_initial_countdown(cooldown):
    return int(cooldown * CALLBACKS_PER_SECOND)


class Callback:
    def __init__(self, squeaks: list, initial_countdown: int, cut_off: float, verbose: bool):
        self._squeaks = squeaks
        self._initial_countdown = initial_countdown
        self._cut_off = cut_off
        self._verbose = verbose

        self._countdown = self._initial_countdown
        self._update_weight = 0.03
        self._old_mean = 0

    def __call__(self, amplitudes: np.ndarray, *_):
        volumes = np.abs(amplitudes)
        current_mean = np.mean(volumes)
        mean_ratio = current_mean / self._old_mean

        if self._verbose:
            typer.echo(f"countdown: {self._countdown}, mean ratio: {mean_ratio:.2f}\r", nl=False)

        if self._countdown == 0 and mean_ratio > self._cut_off:
            squeak_data, squeak_sample_rate = random.choice(self._squeaks)
            sd.play(squeak_data, squeak_sample_rate)

            self._countdown = self._initial_countdown

        self._countdown += -1
        self._countdown = 0 if self._countdown < 0 else self._countdown
        self._old_mean = (1 - self._update_weight) * self._old_mean + self._update_weight * current_mean  # fmt: skip


if __name__ == "__main__":
    app()
