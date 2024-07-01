from setuptools import find_packages, setup

setup(
    name='mypythonlib',
    packages=find_packages(include=['mypythonlib']),
    author='Me',
    version='0.1.1',
    description='My first Python library',
    install_requires=["numpy", "scipy", "matplotlib","h5py"],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
