from setuptools import find_packages, setup

setup(
    name="ferminet_tum",
    version="0.0.1",
    author="Kadir Çeven",
    author_email="kadir.ceven@bilkent.edu.tr",
    description="A library which trains the Fermionic Neural Network to find the ground state wave functions of an atom or a molecule using neural network quantum states.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    license="Apache License 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "jax", "jaxlib", "mendeleev", "chex", "optax", "h5py"],
)
