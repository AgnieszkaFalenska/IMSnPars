from setuptools import setup


def read(fname):
    import os
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='imsnpars',
      version='0.3.0',
      description=' IMS Neural Dependency Parser',
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      url='http://github.com/ulf1/dynet-imsnpars-hdt',
      author='Agnieszka Faleńska',
      author_email='agnieszka.falenska@ims.uni-stuttgart.de',
      license='Apache License 2.0',
      packages=['imsnpars'],
      install_requires=[
          'dynet>=2.0.0',
          'networkx<2.5,>=2.1'
      ],
      scripts=[
          "scripts/imsnpars_downloader.py"
      ],
      python_requires='>=3.6',
      zip_safe=False)
