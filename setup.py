#!/usr/bin/env python

from distutils.core import setup


try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements("requirements.txt", session=False)

# reqs is a list of requirement
try:
    reqs = [str(ir.req) for ir in install_reqs]
except:
    reqs = [str(ir.requirement) for ir in install_reqs]


setup(name='deepaerialmapper',
      version='1.0',
      description='DeepAerialMapper Utilities',
      author='Huijo Kim',
      author_email='ccomkhj@gmail.com',
      packages=['deepaerialmapper'],
      url='https://github.com/RobertKrajewski/DeepAerialMapper',
      install_requires=reqs
     )