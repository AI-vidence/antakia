cd ..
poetry run python -m cProfile -o program.prof dev/run_california.py
snakeviz program.prof