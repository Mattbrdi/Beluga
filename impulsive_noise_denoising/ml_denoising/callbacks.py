from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np
from lightning.pytorch.callbacks import Callback

def mask_edges(mask):
    mask = np.squeeze(np.asarray(mask, dtype=bool))
    padded = np.r_[False, mask, False]
    changes = np.diff(padded.astype(np.int8))
    return zip(np.where(changes == 1)[0], np.where(changes == -1)[0])


def draw_mask(ax, time_axis, mask, color, label):
    for index, (start, end) in enumerate(mask_edges(mask)):
        ax.axvspan(time_axis[start], time_axis[min(end, len(time_axis) - 1)], color=color, alpha=0.25, label=label if index == 0 else None)


class MaskPlotCallback(Callback):
    def on_test_end(self, trainer, pl_module):
        if not pl_module.plot_examples:
            return

        output_path = Path(trainer.logger.log_dir) / "mask_examples.png"
        fig, axes = plt.subplots(len(pl_module.plot_examples), 1, figsize=(13, 2.4 * len(pl_module.plot_examples)), sharex=False)
        axes = np.atleast_1d(axes)

        for ax, example in zip(axes, pl_module.plot_examples):
            signal = np.squeeze(example["signal"].numpy())
            frame_rate = example["frame_rate"]
            time_axis = np.arange(signal.size) / frame_rate

            ax.plot(time_axis, signal, color="black", linewidth=0.5)
            draw_mask(ax, time_axis, example["target"], "tab:red", "target")
            draw_mask(ax, time_axis, example["pred"], "tab:blue", "prediction")
            ax.set_title(example["segment_name"], fontsize=9)
            ax.set_ylabel("amp")
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)