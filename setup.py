from setuptools import setup, find_packages

setup(
    name="holography",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, e.g., "numpy>=1.21.0"
    ],
    extras_require={
        "dev": ["pytest", "black"],  # Development dependencies
    },
    author="Sriram",
    description="A package to create bottlebeams and tweezers",
)