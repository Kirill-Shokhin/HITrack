from setuptools import setup, find_packages
import os

def readme():
  with open('README.md', 'r') as f:
    return f.read()

def _parse_requirements(path):
  with open(path) as f:
    return [
        line.rstrip()
        for line in f
        if not (line.isspace() or line.startswith('#'))
    ]

setup(
  name='HITrack',
  version='0.1.1',
  author='Kirill Shokhin',
  author_email='kashokhin@gmail.com',
  description='3D scene on a monocular video',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/Kirill-Shokhin/HITrack',
  packages=find_packages(),
  install_requires=_parse_requirements('requirements.txt'),
  classifiers=[
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
  keywords='HITrack',
  python_requires='>=3.7'
)