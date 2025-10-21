# GRIP-Tomo 2.0

Updated tooling for the GRIP-Tomo pipeline: convert cryo-ET subtomograms into graphs, extract features, and train machine-learning classifiers. 

> **Note:** This is research software under active development. The software, interfaces, and workflow may change.

## Quick Start
- Clone and enter the repo: `git clone git@github.com:EMSL-Computing/grip-tomo.git && cd grip-tomo`
- Create an isolated environment with uv: `uv venv` (creates `.venv/`)
- Activate the environment: `source .venv/bin/activate`
- Install the package with the extras that fit your hardware:
	- CPU-only baseline: `uv pip install .`
	- Parsl workflow support: `uv pip install .[parsl]`
	- GPU (RAPIDS) stack: `uv pip install .[gpu-cu11]` or `uv pip install .[gpu-cu12]`
	- Editable mode for active development: `uv pip install -e ."[dev]"`
- Run smoke tests when ready: `uv run pytest -q tests/test_core.py`

## Docs & Tutorials
- API docs (`pdoc`): `docs/api/index.html`
- Subtomogram simulation quickstart: `docs/cistem_subtomogram_simulation_quickstart.md`
- Tutorial notebook (view in VS Code/Jupyter): `docs/tutorial_notebook/tutorial.ipynb`
- Parsl on HPC quickstart: `docs/parsl_hpc_quickstart.md`

## Contributing
- Open an issue or discuss the plan before large changes.
- Fork the repo, create a feature branch, and keep commits focused.
- Document new functionality and add tests; run `uv run pytest` before submitting.
- Regenerate docs when APIs change: `uv run pdoc griptomo --docformat numpy --output-directory docs/api`.
- Submit a merge request describing the change and verification steps.

### Citation
Please cite our publication and accompanying software if you use it:

> George, A, Kim, DN, Moser, T, Gildea, IT, Evans, JE, Cheung, MS. Graph identification of proteins in tomograms (GRIP-Tomo). Protein Science. 2023; 32( 1):e4538. https://doi.org/10.1002/pro.4538

> George, A, Kim, DN, Moser, T, Gildea, IT, Evans, JE, Cheung, MS. EMSL-Computing/grip-tomo: Version 1.0. Zenodo; 2023. https://doi.org/10.5281/zenodo.17127842 


---

See the included license and disclaimer files

