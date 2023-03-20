from setuptools import setup
from setuptools import find_namespace_packages

setup(
    name='rlsdk',
    version='0.0.1',
    description='Software Development Kit for RLFrame',
    author='zhizunjiege',
    author_email='cjason@buaa.edu.cn',
    keywords='rl,grpc,rest',
    platforms='linux,macos,windows',
    license='MIT',
    zip_safe=True,
    python_requires='>=3.8',
    install_requires=[
        'grpcio>=1.47.0',
        'grpcio-tools>=1.47.0',
        'numpy>=1.23.2',
        'requests>=2.28.1',
    ],
    packages=find_namespace_packages(where='src', exclude=['tests']),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        'rlsdk.schemas': ['**/*.json']
    },
)