"""Where PINTHAC's generated data lives.

Why this exists: several modules read files that are *generated* rather than authored --
the supercritical-water property lookup table, the DeepONet training sets, the trained
network checkpoints. Those are large, regenerable, and deliberately kept out of version
control, so they cannot sit beside the source files that read them the way the IAPWS
coefficient tables do. One module knowing the directory keeps that decision in a single
place instead of a `script_dir` in every consumer.

Set the PINTHAC_DATA environment variable to point somewhere else -- a scratch volume, a
shared cache -- without editing any code.
"""
import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)

DATA_DIR = os.environ.get("PINTHAC_DATA", os.path.join(_REPO_ROOT, "data"))


def data_file(name):
    """Absolute path to a generated data file.

    Inputs:
        name : file name relative to the data directory, string
    Returns:
        path : absolute path, string. The file is not required to exist -- a generator
               writing a new table needs the path before the file does.
    """
    return os.path.join(DATA_DIR, name)
