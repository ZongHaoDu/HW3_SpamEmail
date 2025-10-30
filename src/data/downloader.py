import os
import urllib.request


def download_file(url: str, out_path: str, force: bool = False) -> str:
    """Download a file from `url` to `out_path`. Returns the path to the file.

    Uses urllib to avoid extra dependencies. Creates parent directories as needed.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and not force:
        return out_path

    with urllib.request.urlopen(url) as resp:
        content = resp.read()
    with open(out_path, "wb") as f:
        f.write(content)
    return out_path


def ensure_data(url: str, out_path: str = "data/raw.csv", force: bool = False) -> str:
    """Ensure a local copy of the dataset exists and return its path."""
    return download_file(url, out_path, force=force)
