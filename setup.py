from setuptools import setup

__version__ = (0, 0, 0)

setup(
    name="crop_classification",
    description="Crop classification using SAR and optical satellite data",
    version=".".join(str(d) for d in __version__),
    author="crop_classification Lim",
    author_email="sangwon3@ualberta.ca",
    packages=["crop_classification"],
    include_package_data=True,
    scripts="""
        ./scripts/softmax
        ./scripts/CNN
    """.split(),
)
