# coding=utf-8
from setuptools import setup, find_packages

setup(name='jitsdp',
      version='0.1.0',
      license='',
      description='jit-sdp-nn',
      url='',
      platforms='Linux',
      classifiers=[
          'Programming Language :: Python :: 3',
      ],
      author='',
      author_email='',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'jitsdp = jitsdp.__main__:main'
          ]
      },
      install_requires=['numpy==1.17.2',
                        'pandas==1.0.1',
                        'torch==1.4.0',
                        'scikit-learn==0.22',
                        'scikit-multiflow==0.5.3',
                        'matplotlib==3.1.2',
                        'seaborn==0.9.0',
                        'mlflow==1.9.1',
                        ],
      python_requires='>=3.7',
      zip_safe=True)
