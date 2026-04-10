from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned"
DATA_RESULTS = PROJECT_ROOT / "data" / "results"
DATA_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings"
DATA_VALIDATION = PROJECT_ROOT / "data" / "validation"
MAT2VEC_MODELS = PROJECT_ROOT / "data" / "mat2vec_models"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MAT2VEC_PRETRAINED = PROJECT_ROOT / "mat2vec" / "mat2vec" / "training" / "models" / "pretrained_embeddings"

# ── Journal plot style (applied globally on import) ──────────────────────────
JOURNAL_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 1.2,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "legend.fontsize": 11,
}
mpl.rcParams.update(JOURNAL_RC)


def journal_savefig(fig_or_path, path=None):
    """Save a figure with journal-mandated settings.

    Usage:
        journal_savefig("plot.png")          # saves current figure
        journal_savefig(fig, "plot.png")     # saves specific figure
    """
    if path is None:
        path = fig_or_path
        plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    else:
        fig_or_path.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
