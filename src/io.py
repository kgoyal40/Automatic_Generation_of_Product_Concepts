from pathlib import Path

# Main Directories
ROOT = Path(__file__).absolute().parent.parent
DATA_DIR = ROOT / "data"

CONTEXT_DATA_DIR = DATA_DIR / "contexts"
CONTEXT_NAMES_FP = DATA_DIR / "context_names_flanders.csv"

ONEHOT_SONG_ID_FP = DATA_DIR / "songs_binary.csv"
ENCODING_FP = DATA_DIR / "attribute-encoding.toml"
