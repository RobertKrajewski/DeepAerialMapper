#!/usr/bin/env python

import pathlib

from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


with pathlib.Path("requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement) for requirement in parse_requirements(requirements_txt)
    ]

setup(
    name="deepaerialmapper",
    version="1.0",
    description="Automated HD Map Creation from Aerial Segmentation Masks",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="DeepAerialMapper Contributors",
    author_email="ccomkhj@gmail.com",
    keywords="computer vision, HD map",
    packages=find_packages(exclude=("configs", "tools", "data", "docs")),
    url="https://github.com/RobertKrajewski/DeepAerialMapper",
    license="GPLv3",
    install_requires=install_requires,
)
