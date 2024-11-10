from typing import List
from pathlib import Path


def load_inputs(restrict_ids: List[str] = None):
    """
    Read from /input/features
    """
    features_list = sorted([fp for fp in Path("/input/features").glob("*.pt")])
    print(f"{len(features_list)} features found under /input/features")
    if restrict_ids is not None:
        features_list = [fp for fp in features_list if fp.stem in restrict_ids]
        print(f"Restricted input to {len(features_list)} cases based on the provided restrict list ({len(restrict_ids)})")
    return features_list