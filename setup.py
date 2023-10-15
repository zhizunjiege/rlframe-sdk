from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name='rlsdk',
    version='1.0.0',
    description='Software Development Kit for RLFrame',
    author='zhizunjiege',
    author_email='cjason@buaa.edu.cn',
    keywords='rl,grpc,rest',
    platforms='linux,macos,windows',
    license='MIT',
    zip_safe=True,
    python_requires='>=3.8',
    install_requires=[
        'grpcio>=1.48.2',
        'grpcio-tools>=1.48.2',
        'numpy>=1.23.2',
    ],
    packages=find_namespace_packages(where='src', exclude=['tests']),
    package_dir={'': 'src'},
)
