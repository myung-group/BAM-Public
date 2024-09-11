from setuptools import find_packages, setup

setup(
    name="BAM",
    version="0.1.0.dev0",
    author="Myung Group",
    author_email="cwmyung@skku.edu",
    description="Uncertainty Quantification of Bayesian Deep Learning ab initio Potential",
    url="https://github.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        #Operating System :: OS Independent",
    ],
    install_requires=[
        "optax",
        "flax",
        "ase",
        "matscipy",
        "jraph",
        "e3nn-jax",
        "e3nn",
        "dm-haiku",
    ],
)



