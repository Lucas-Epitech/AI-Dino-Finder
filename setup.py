from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""

try:
    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = []
        for line in f:
            if line.strip() and not line.startswith("#"):
                requirements.append(line.strip())
#       requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    requirements = []

setup(
    # Metadata about your package
    name='dino-finder',
    version='0.1.0',
    author='Lucas Carré',
    description='A tool to find dinosaurs in images using machine learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # Config
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    # Dependencies
    install_requires=requirements,

    # Versionning
    python_requires='>=3.11',
)
