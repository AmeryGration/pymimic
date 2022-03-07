#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="pymimic",
    version="0.1.0",
    author="Amery Gration",
    author_email="amerygration@hotmail.com",
    description="A pure-Python implementation of linear prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmeryGration/pymimic",
    package_data={"pymimic": ["design_data/*.dat"]},
    packages=setuptools.find_packages(),
    python_requires=">3.6.0",
    install_requires=["numpy", "scipy", "matplotlib"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU general public license (GPL)",
        "Operating System :: OS Independent",
    ),
)
