from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="riskaverse_DP",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,  # <- use requirements.txt
    author="Dr. Matilde Gargiani",
    description="Semismooth Newton methods for risk-averse MDPs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.8',
)
