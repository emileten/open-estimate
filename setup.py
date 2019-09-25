from setuptools import setup, find_packages

setup(name='openest',
      version='0.1',
      description='Library of empirical model application.',
      url='http://github.com/jrising/open-estimate',
      author='James Rising',
      author_email='jarising@gmail.com',
      license='GNU v. 3',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'emcee', 'statsmodels', 'xarray',
                        'metacsv'],
      tests_require=['pytest', 'pytest-mock'],
      zip_safe=False,
      )
