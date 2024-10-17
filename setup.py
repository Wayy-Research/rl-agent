from setuptools import setup, find_packages

setup(
    name='rl_agent',
    version='0.1.0',
    packages=find_packages(where='rl_agent'),
    package_dir={'': 'rl_agent'},
    install_requires=[
        'numpy',
        'pandas',
        'gym',
        'tensorflow',
        'torch'
    ],
    author='Rick Galbo',
    author_email='rcgalbo@gmail.com',
    description='A reinforcement learning-based trading package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rcgalbo/rl-agent',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
