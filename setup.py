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
                        'pandas==0.25.1',
                        'torch==1.3.1',
                        'scikit-learn==0.22',
                        ],
      python_requires='>=3.7',
      zip_safe=True)
