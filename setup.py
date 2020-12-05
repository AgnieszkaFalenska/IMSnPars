from setuptools import setup
import m2r


def get_readme():
    return m2r.parse_from_file('README.md')


setup(name='imsnpars',
      version='0.3.4',
      description=' IMS Neural Dependency Parser',
      long_description=get_readme(),
      long_description_content_type='text/x-rst',
      url='https://github.com/AgnieszkaFalenska/IMSnPars',
      author='Agnieszka Faleńska',
      author_email='agnieszka.falenska@ims.uni-stuttgart.de',
      license='Apache License 2.0',
      packages=['imsnpars'],
      install_requires=[
          'dynet>=2.0.0',
          'networkx<2.5,>=2.1',
          'h5py>=3.1.0'
      ],
      scripts=[
          "scripts/imsnpars_downloader.py"
      ],
      python_requires='>=3.6',
      zip_safe=False)
