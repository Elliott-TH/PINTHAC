# Contributing

Install the development dependencies with `pip install -e ".[test]"` and run
`python -m pytest`. Optional benchmark and training dependencies are available
through the `benchmark` and `ml` extras.

Keep numerical changes separate from documentation or formatting changes. Include
regression tests for bug fixes and cite the source of new correlations and constants.
Document input/output units, valid ranges, uncertainty, and numerical limitations.
Use kelvin for temperature and specify whether pressure is in MPa or Pa.

Follow the existing function-based interfaces. Preserve NumPy and PyTorch behavior,
including gradients where supported. Keep comments focused on numerical choices
and physics that are not apparent from the code.

Submit a pull request describing the change and the checks you ran. Do not include
reference PDFs, private data, credentials, generated datasets, or model checkpoints.
