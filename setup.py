
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="powerbox_jax",
    version="0.10",
    author="Lucas Makinen",
    author_email="timothy.makinen@cfa.harvard.edu",
    description="Jax implementation of Stephen Murray's powerbox\
    in Jax.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tlmakinen/powerbox-jax.git",
    packages=["powerbox_jax"],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
    install_requires=[
          "jax>=0.2.21",
          "numpyro>=0.4.1",
          "tqdm>=4.48.2",
          "numpy>=1.16.0",
          "scipy>=1.4.1",
          "corner",
          "matplotlib"],
)
