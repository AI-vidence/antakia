import setuptools

setuptools.setup(
    name="antakia",
    version="0.1",
    author="Antoine EDY AI-vidence (c)",
    author_email="antoineedy@outlook.fr",
    description="XAI made simple",
    long_description=open("README.md").read(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
)
