from pathlib import Path
import upet


def ensure_model(model_path: Path, *, version: str, size: str) -> Path:
    """Export PET-MAD with upet when model_path is not present."""
    model_path = model_path.expanduser().resolve()
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting PET-MAD ({size}, version {version}) to {model_path}")
    upet.save_upet(model="pet-mad", size=size, version=version, output=str(model_path))
    return model_path
