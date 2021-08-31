from setuptools import setup

version = {}
with open("gustav/version.py") as f:
    exec(f.read(), version)

setup(
    name="gustav",
    version=version,
    desription="Fast geometry prototyper.",
    author="Jaewook Lee",
    author_email="jlee@ilsb.tuwien.ac.at",
    packages=["gustav"],
    install_requires=[
        "tetgen",
        "splinelibpy",
        "triangle"
    ],
)
