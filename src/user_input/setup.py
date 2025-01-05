from setuptools import setup, find_packages

setup(
    name='user_input',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'torch',  # Add other dependencies if necessary
        'numpy'
    ],
    include_package_data=True,
    zip_safe=False,
)
