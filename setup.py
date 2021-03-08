'''setup file of the package'''

import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='hmnn',
    version='0.0.1',
    author='Thomas Mathieu',
    author_email='thomas.mathieu@m4x.org',
    description='Personal deep learning implementation inspired by keras',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/bdepebhe/handmade-neural-network',
    packages=[
        'hmnn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'matplotlib>=3.3.2',
        'numpy>=1.18.4',
        'pandas>=1.1.4',
        'scikit-learn>=0.23.2',
        'seaborn>=0.11.0',
        'tensorflow==2.3.1',
    ]
)
