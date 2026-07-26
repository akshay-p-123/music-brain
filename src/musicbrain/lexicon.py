"""NRC-VAD lexicon.

The NRC-VAD Lexicon (Mohammad, 2018/2025) is licensed for non-commercial
research use but its "No Redistribution" term forbids checking the raw
file into version control. This module downloads it on demand into
``data/lexicons/`` (gitignored) rather than shipping it in the repo.

This is a dataset that maps a bunch of words to VAD (valence-arousal-dominance) ratings (-1 to 1).
valence: positive/negative sentiment
arousal: how exciting a word is
dominance: the level of power a word has

NRC-VAD is used to _____
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

NRC_VAD_URL = "http://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"
DEFAULT_LEXICON_DIR = Path(__file__).resolve().parents[2] / "data" / "lexicons"
UNIGRAM_RELATIVE_PATH = (
    "NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt"
)


def fetch_nrc_vad(dest_dir: Path = DEFAULT_LEXICON_DIR, force: bool = False) -> Path:
    #Download and extract the NRC-VAD lexicon into dest_dir.

    dest_dir = Path(dest_dir)
    unigram_path = dest_dir / UNIGRAM_RELATIVE_PATH
    if unigram_path.exists() and not force:
        return unigram_path

    dest_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "curl/8.0.1"}
    response = requests.get(NRC_VAD_URL, timeout=60, headers=headers)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(dest_dir)

    if not unigram_path.exists():
        raise FileNotFoundError(
            f"Expected {unigram_path} after extraction; lexicon layout may have changed."
        )
    return unigram_path


class NRCVAD:
    """Lookup table mapping a word to its (valence, arousal, dominance) score.

    Scores are real-valued in roughly [-1, 1] (NOT [0, 1] -- verify against
    the README of whichever copy you fetch before assuming otherwise).
    """

    def __init__(self, unigram_path: Path | None = None):
        path = Path(unigram_path) if unigram_path is not None else fetch_nrc_vad()
        table = pd.read_csv(path, sep="\t")
        table = table.set_index("term")
        self._table = table

    def __contains__(self, word: str) -> bool:
        return word in self._table.index

    def va(self, word: str) -> tuple[float, float]:
        """Return (valence, arousal) for *word*, dropping dominance."""
        row = self._table.loc[word]
        return float(row["valence"]), float(row["arousal"])

    def vad(self, word: str) -> tuple[float, float, float]:
        row = self._table.loc[word]
        return float(row["valence"]), float(row["arousal"]), float(row["dominance"])
