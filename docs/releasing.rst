Releasing
=========

Run ``uv sync --frozen --extra docs``, ``uv run pytest``,
``uv run sphinx-build -W -b html docs docs/_build/html``, and ``uv build``.
The build creates both a wheel and a source distribution under ``dist/``.
Inspect the version in ``pyproject.toml`` and the distribution contents before
creating a release.

The local ``python-publish.yml`` workflow uses PyPI trusted publishing when a
GitHub release is published. Before the first release, configure a PyPI
trusted publisher for this repository, workflow ``python-publish.yml``, and
environment ``pypi``. Creating a release and configuring that publisher are
separate remote actions; neither is performed by the library build.
