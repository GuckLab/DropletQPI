from distutils.core import setup


author = "Timon Beck"
authors = [author]
description = 'Segment spherical samples from a phase image ' \
              'and evaluate the refractive index'
name = 'DropletQpi'
year = "2021"

setup(
    name=name,
    author=author,
    author_email='timon.beck@mpl.mpg.de',
    url='https://github.com/GuckLab/DropletQPI',
    include_package_data=True,
    license="None",
    description=description,
    install_requires=["h5py>=2.10.0",
                      "numpy>=1.17.0",
                      "matplotlib",
                      "qpsphere",
                      "qpimage",
                      "drymass",
                      "scikit-image>=0.17.2",
                      "scipy>=0.12.0",
                      "pandas",
                      ],
    python_requires=">=3.6",
    keywords=["image analysis", "biology", "microscopy"],
)
