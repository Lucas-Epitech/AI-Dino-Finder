"""
Setup configuration for the dino_finder package.

This file allows installing the package in "editable" mode:
    pip install -e .

This way you can import from anywhere:
    from dino_finder.models import DinoClassifier
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="dino-finder",
    version="0.1.0",
    author="Lucas",
    description="Dinosaur species classification using CNNs with PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Lucas-Epitech/AI-Dino-Finder",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
