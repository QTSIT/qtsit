"""Python Setup.py file for qtist package."""
import sys
import time
from setuptools import setup, find_packages

if '--release' in sys.argv:
    IS_RELEASE = True
    sys.argv.remove('--release')
else:
    # Build a nightly package by default.
    IS_RELEASE = False


# get the version from qtsit/__init__.py
def _get_version():
    with open('qtsit/__init__.py') as fp:
        for line in fp:
            if line.startswith('__version__'):
                g = {}
                exec(line, g)
                base = g['__version__']
                if IS_RELEASE:
                    return base
                else:
                    # nightly version : .devYearMonthDayHourMinute
                    if base.endswith('.dev') is False:
                        # Force to add `.dev` if `--release` option isn't passed when building
                        base += '.dev'
                    return base + time.strftime("%Y%m%d%H%M%S")

        raise ValueError('`__version__` not defined in `qtsit/__init__.py`')


setup(name='qtsit',
      version=_get_version(),
      url='https://github.com/qtsit/qtsit',
      maintainer='QTSIT contributors',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Information Technology',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
      ],
      license='MIT',
      description='QTSIT: Quantum Technologies for Industry Transformation Toolkit',
      keywords=[
          'qtsit',
          'quantum-chemistry',
		  'quantum-computing',
		  'quantum-ml',
		  'quantum-ai',
      ],
      packages=find_packages(exclude=["*.tests"]),
      project_urls={
          'Documentation': 'https://qtsit.readthedocs.io/en/latest/',
          'Source': 'https://github.com/QTSIT/qtsit',
      },
      install_requires=[
          'joblib',
          'numpy',
      ],
      python_requires='>=3.9,<3.12')