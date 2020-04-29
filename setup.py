import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sghmc-2song",
    version="0.0.2",
    author="Jiajun Song; Yiping Song",
    author_email="jiajun.song@duke.edu",
    description="A package for SGHMC, STA663 Final Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JiajunSong629/SGHMC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)