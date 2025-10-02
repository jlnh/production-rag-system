"""
Setup script for RAG Production System
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
            and not line.startswith("#")
            and not line.startswith("-r")
        ]

# Base requirements
install_requires = read_requirements("requirements.txt")

# Development requirements (optional)
dev_requires = []
if os.path.exists("requirements-dev.txt"):
    dev_requires = read_requirements("requirements-dev.txt")

setup(
    name="rag-production-system",
    version="1.0.0",
    author="RAG Course",
    author_email="contact@ragcourse.com",
    description="Production-ready Retrieval-Augmented Generation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/rag-production-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "test": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "httpx>=0.25.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-server=rag_system.api.main:main",
            "rag-cli=rag_system.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rag_system": [
            "config/*.yml",
            "config/*.json",
            "templates/*.html",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-org/rag-production-system/issues",
        "Source": "https://github.com/your-org/rag-production-system",
        "Documentation": "https://rag-production-system.readthedocs.io/",
    },
)