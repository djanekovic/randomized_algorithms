from setuptools import setup

setup(
    name='randomized_algorithms',
    version='0.0.1',
    packages=['randomized_algorithms'],
    install_requires=[
        'numpy',
        'scipy',
        'sklearn'
    ],
    extras_require= {
        'gpu': 'cupy'
    },

    author="Darko Janekovic",
    author_email="darko.janekovic@gmail.com",
    license="MIT",
)
