import pathlib
import setuptools

lines = pathlib.Path("requirements.txt").read_text()
requirements = [line for line in lines.split("\n") if line != "" and not line.startswith("git+")]

requirements += ["mkdocstrings-python", "mkdocs-material", "mkdocs", "mkdocstrings"]

git_r = [line for line in lines.split("\n") if line != "" and line.startswith("git+")]
for git in git_r:
    git = 'skope-rules @ ' + git
    requirements.append(git)

setuptools.setup(
    name="antakia",
    version="0.1",
    author="AI-vidence (c)",
    description="XAI made simple",
    long_description=open('DESCRIPTION.rst').read(),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
)