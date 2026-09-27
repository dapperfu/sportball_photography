#!/usr/bin/env python3
"""
Setup script for sportball package using versioneer for version management.
"""

try:
    import versioneer
except ImportError:
    # Fallback for when versioneer is not available
    class MockVersioneer:
        @staticmethod
        def get_version():
            return "1.3.4"

        @staticmethod
        def get_cmdclass():
            return {}

    versioneer = MockVersioneer()

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sportball",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Sportball Team",
    author_email="team@sportball.ai",
    description="Sports photo organization: EXIF game splitting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sportball/sportball",
    project_urls={
        "Bug Reports": "https://github.com/sportball/sportball/issues",
        "Source": "https://github.com/sportball/sportball.git",
        "Documentation": "https://sportball.readthedocs.io",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Image helpers
        "Pillow>=10.0.0",
        # CLI and UI
        "click>=8.1.0",
        "rich>=13.0.0",
        "tqdm>=4.65.0",
        "loguru>=0.7.0",
        # Data Processing
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "pathspec>=0.11.0",
        "psutil>=5.9.0",
        "fast-exif-rs-py @ git+https://github.com/dapperfu/fast-exif-rs-py.git#egg=fast-exif-rs-py",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
        "all": [
            "sportball[dev]",
        ],
    },
    entry_points={
        "console_scripts": [
            "sportball=sportball.cli.main:main",
            "sb=sportball.cli.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
