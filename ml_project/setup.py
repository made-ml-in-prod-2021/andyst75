from setuptools import find_packages, setup


def parse_requirements(path):
    with open(path) as fin:
        requirements = []
        for line in fin:
            requirements.append(line.strip())

    return requirements


setup(
    name="ml_example",
    packages=find_packages(),
    version="0.1.0",
    description="Homework1 on ML in production",
    author="AndreW Starikov",
    install_requires=parse_requirements("./requirements/runtime.txt"),
    license="MIT",
)
