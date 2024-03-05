cd ../antakia-core
poetry install
poetry lock
poetry build
poetry publish

cd ../antakia
poetry install
poetry lock
poetry build
poetry publish