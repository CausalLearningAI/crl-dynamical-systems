from setuptools import find_packages, setup

setup(
    name="mech-nn",
    version="0.1.0",
    # install_requires=['Your-Library'],
    packages=find_packages("mech-nn"),
    package_dir={"": "mech-nn"},
    # url='https://github.com/your-name/your_app',
    # license='MIT',
    # author='Your NAME',
    # author_email='your@email.com',
    description="Mechnistic Neural Networks",
)
