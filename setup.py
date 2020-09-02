from setuptools import setup, find_packages

setup(name='openest',
      use_scm_version=True,
      description='Library of empirical model application.',
      url='http://github.com/jrising/open-estimate',
      author='James Rising',
      author_email='jarising@gmail.com',
      license='GNU v. 3',
      packages=find_packages(),
      setup_requires=['setuptools_scm'],
      install_requires=['numpy', 'scipy', 'emcee', 'statsmodels', 'bottleneck',
                        'xarray', 'pandas', 'metacsv'],
      extras_require={
            "test": ["pytest", "pytest-mock"],
      },
      zip_safe=False,
      )
