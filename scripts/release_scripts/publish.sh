cd ../antakia-core
poetry shell
poetry install
poetry build
poetry publish

cd ../antakia
poetry shell
poetry install
poetry build
poetry publish