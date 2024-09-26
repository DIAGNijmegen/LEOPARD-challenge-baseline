from pathlib import Path

def load_inputs():
    """
    Read from /input/features
    """
    features_list = sorted([fp for fp in Path("/input/features").glob("*.pt")])
    return features_list