from setuptools import setup, find_packages

setup(
    name='Expanda',
    version='1.3.1',

    author='Jungwoo Park',
    author_email='affjljoo3581@gmail.com',

    description='Integrated Corpus-Building Environment',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',

    keywords=['expanda', 'corpus', 'dataset', 'nlp'],
    url='https://github.com/affjljoo3581/Expanda',
    license='Apache-2.0',

    package_dir={'': 'src'},
    packages=find_packages('src'),
    python_requires='>=3.10.0',
    install_requires=[
        'PyYAML==6.0.2',
        'nltk',
        'ijson',
        'tqdm',
        'mwparserfromhell',
        'tokenizers',
        'datasets',
        'kss',
        'huggingface_hub',
        'python-dotenv'
    ],

    entry_points={
        'console_scripts': [
            'expanda = expanda:_main',
            'expanda-shuffling = expanda.shuffling:_main',
            'expanda-tokenization = expanda.tokenization:_main'
        ]
    },

    classifiers=[
        'Environment :: Console',
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
