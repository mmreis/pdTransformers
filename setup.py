from setuptools import setup

setup(name='pdTransformers',
      version='0.1.0',
      description='TransformerMixin for Pipeline building',
      url='https://github.com/mmreis/pdTransformers',
      author='Marisa Reis',
      license='MIT',
      packages=['pdTransformers'],
      install_requires=[
          'pandas',
          'numpy',
          'scikit-learn',
      ],
      zip_safe=False)