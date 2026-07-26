"""Vertex -> Schaefer-400 parcellation for TRIBEv2's fsaverage5 output.
We are grouping TRIBEv2 predicted ~20k vertices -> 400 canonical regions across 7 networks.

TRIBEv2 predicts on the fsaverage5 cortical mesh (~20,484 vertices, two
hemispheres of 10,242 each). The "5" is the level of brain map resolution of the TRIBEv2 output.
This module aggregates that raw vertex output down to the trunk's assumed 400 vertex input using the
canonical Schaefer-400 (7-network) FreeSurfer annotation for fsaverage5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import requests

CBIG_BASE_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/"
    "FreeSurfer5.3/fsaverage5/label"
)
DEFAULT_ATLAS_DIR = Path(__file__).resolve().parents[2] / "data" / "atlases"
N_PARCELS = 400
N_NETWORKS = 7

#utils
def _annot_filename(hemi: str) -> str:
    return f"{hemi}.Schaefer2018_{N_PARCELS}Parcels_{N_NETWORKS}Networks_order.annot"


def fetch_schaefer400_fsaverage5(dest_dir: Path = DEFAULT_ATLAS_DIR) -> dict[str, Path]:
    """Download the lh/rh Schaefer-400 fsaverage5 annot files if not already cached."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for hemi in ("lh", "rh"):
        fname = _annot_filename(hemi)
        path = dest_dir / fname
        if not path.exists():
            resp = requests.get(f"{CBIG_BASE_URL}/{fname}", timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        paths[hemi] = path
    return paths


class Schaefer400Parcellator:
    """Aggregates fsaverage5 vertex data to Schaefer-400 parcels by averaging.
    """

    def __init__(self, atlas_dir: Path = DEFAULT_ATLAS_DIR):
        from nibabel.freesurfer.io import read_annot

        paths = fetch_schaefer400_fsaverage5(atlas_dir)
        hemi_labels = {}
        hemi_names = {}
        for hemi, path in paths.items():
            labels, _, names = read_annot(str(path)) #which parcel, _, human readable name of each parcel
            hemi_labels[hemi] = labels  # (n_vertices_per_hemi,) parcel index per vertex
            hemi_names[hemi] = [n.decode("utf-8") for n in names]

        n_left = len(hemi_labels["lh"])
        n_right = len(hemi_labels["rh"])
        self.n_vertices = n_left + n_right
        concat_labels = np.concatenate(
            [hemi_labels["lh"], hemi_labels["rh"] + len(hemi_names["lh"])]
        )
        all_names = hemi_names["lh"] + hemi_names["rh"]

        # Each hemisphere's .annot carries one extra non-cortical label
        # (unlabeled / medial-wall vertices) alongside its 200 real
        # Schaefer parcels -- confirmed empirically: read_annot on these
        # files reports 402 labels total, not 400, and index 0 of each
        # hemisphere is named "Background+FreeSurfer_Defined_Medial_Wall".
        # Excluded here so aggregate() returns exactly the spec's P=400,
        # not 402 with two meaningless all-background columns.
        keep = [i for i, n in enumerate(all_names) if "background" not in n.lower()]
        if len(keep) != N_PARCELS:
            raise RuntimeError(
                f"Expected {N_PARCELS} non-background parcels after filtering, "
                f"got {len(keep)} -- the annot file's label layout may differ "
                "from what this filter assumes; inspect `all_names` directly."
            )
        self.parcel_names = [all_names[i] for i in keep]
        self.n_parcels = len(self.parcel_names)

        self._parcel_vertex_masks = [concat_labels == i for i in keep] #lookup table of labeled vertices to be used in aggregate

    def aggregate(self, vertex_data: np.ndarray) -> np.ndarray:
        """Average vertex-level predictions within each parcel.

        Parameters
        ----------
        vertex_data:
            Array of shape (n_timesteps, n_vertices) matching TRIBEv2's
            `preds` output, or (n_vertices,) for a single timestep.

        Returns
        -------
        Array of shape (n_timesteps, n_parcels), or (n_parcels,).
        """
        vertex_data = np.asarray(vertex_data)
        single_timestep = vertex_data.ndim == 1
        if single_timestep:
            vertex_data = vertex_data[None, :]

        if vertex_data.shape[-1] != self.n_vertices:
            raise ValueError(
                f"Expected {self.n_vertices} vertices (fsaverage5, both hemispheres), "
                f"got {vertex_data.shape[-1]}. Background/unlabeled vertices "
                "('Medial_wall') are included in this count and will simply "
                "average into a low-signal parcel."
            )

        n_timesteps = vertex_data.shape[0]
        out = np.zeros((n_timesteps, self.n_parcels), dtype=np.float64)
        for i, mask in enumerate(self._parcel_vertex_masks):
            if mask.any():
                out[:, i] = vertex_data[:, mask].mean(axis=-1)
        return out[0] if single_timestep else out
