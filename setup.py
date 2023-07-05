import pathlib
import setuptools

lines = pathlib.Path("requirements.txt").read_text()
requirements = [line for line in lines.split("\n") if line != ""]

requirements += ["mkdocstrings-python", "mkdocs-material", "mkdocs", "mkdocstrings"]

setuptools.setup(
    name="antakia",
    version="0.1",
    author="Antoine EDY AI-vidence (c)",
    author_email="antoineedy@outlook.fr",
    description="XAI made simple",
    long_description=open("README.md").read(),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
)
