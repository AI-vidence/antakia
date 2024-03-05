cd ../antakia-core
poetry shell
poetry install
poetry build
poetry publish

cd ../antakia-ac
poetry shell
poetry install
poetry build
poetry publish