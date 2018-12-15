from setuptools import setup

setup(
    name='FEMpy',
    version='1.0',
    packages=['FEMpy'],
    install_requires=[
        'numpy',
        'scipy',
    ],
    url='https://github.com/floydie7/FEMpy',
    license='MIT',
    author='Benjamin Floyd',
    author_email='benjaminfloyd7@gmail.com',
    description='A Finite Element Method solver'
)
