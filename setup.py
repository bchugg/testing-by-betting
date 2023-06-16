from setuptools import setup

# load long description from README
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='testing-by-betting',
    version='0.1',
    description='Nonparametric sequential hypothesis testing by betting',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/bchugg/testing-by-betting',
    author='Ben Chugg',
    author_email='benchugg@cmu.edu',
    license='MIT',
    packages=['testing_by_betting'],
    zip_safe=False
)