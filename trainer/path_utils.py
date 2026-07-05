from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def resolve_project_paths(args, *names):
    for name in names:
        value = getattr(args, name, None)
        if value:
            setattr(args, name, project_path(value))
    return args
