from setuptools import setup, find_packages

setup(
    name="sugar",
    version="0.1.0",
    description="SUGAR: Socially-grounded, Uncertainty-Gated, Augmented Recommender",
    packages=find_packages(include=["sugar", "sugar.*"]),
    python_requires=">=3.10",
)