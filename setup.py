from setuptools import setup, find_packages

setup(
    name="lumenspark",
    version="0.1.5",
    description="Lumenspark: A Transformer Model Optimized for Text Generation and Classification with Low Compute and Memory Requirements.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Anthony Therrien",
    url="https://github.com/anto18671/lumenspark",
    packages=find_packages(),
    license="MIT",
    keywords=["transformers", "deep learning", "NLP", "PyTorch", "machine learning"],
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Source": "https://github.com/anto18671/lumenspark",
        "Documentation": "https://github.com/anto18671/lumenspark/blob/main/README.md",
        "Bug Tracker": "https://github.com/anto18671/lumenspark/issues",
    },
)
