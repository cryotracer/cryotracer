# Installation

Use Python 3.11–3.14 and Git. Create a virtual environment and install
CryoTracer directly from GitHub:

```sh
python3 -m venv ~/cryotracer-venv
source ~/cryotracer-venv/bin/activate
python -m pip install "git+https://github.com/cryotracer/cryotracer.git"
cryotracer --help
```

Activate the same environment when you return to use CryoTracer:

```sh
source ~/cryotracer-venv/bin/activate
```

Continue with [prediction](prediction.md) using a provided CryoTracer checkpoint.
