from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import typer

app = typer.Typer()

countdown = 0
old_mean = 0


@app.command()
def run(
    source: Optional[str] = typer.Argument(
        None,
        help="Audios source path. If none, use built-in",
    ),
    cooldown: int = typer.Option(
        1_000,
        help="Wait at least --cooldown seconds before emiting another sound",
    ),
    cut_off: float = typer.Option(
        10,
        help="How much INsensible RatinhoNet should be to sudden volume changes",
    ),
    log: bool = typer.Option(False),
):
    """
    Run RatinhoNet with SOURCE audios.

    Audios should be short (< 5 seconds) WAV files.
    """

    update_weight = 0.03
    channels = 1

    global countdown, old_mean

    countdown = cooldown
    old_mean = 0
    ratinho_sound, ratinho_sample_rate = sf.read("audio/fart.wav")

    def callback(amplitudes, frames, time, status):
        global countdown, old_mean

        volumes = np.abs(amplitudes)
        current_mean = np.mean(volumes)
        mean_ratio = current_mean / old_mean

        if log:
            typer.echo(f"countdown: {countdown}, mean ratio: {mean_ratio:.2f}\r", nl=False)

        if countdown == 0 and mean_ratio > cut_off:
            countdown = cooldown
            sd.play(ratinho_sound, ratinho_sample_rate)

        countdown -= 1
        countdown = 0 if countdown < 0 else countdown
        old_mean = (1 - update_weight) * old_mean + update_weight * current_mean

    with sd.InputStream(channels=channels, callback=callback):
        input()


if __name__ == "__main__":
    app()
