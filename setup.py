#!/usr/bin/env python3
"""Setup script for Advanced AutoML System."""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="automl",
    version="0.1.0",
    author="Frank Hutter Fan Club",
    author_email="moeinghaeini@gmail.com",
    description="Advanced AutoML SS25 Exam - Tabular with Multi-Fidelity Optimization and Meta-Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/automl-exam-ss25-tabular-frankhutterfanclub-1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "automl-run=run:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 