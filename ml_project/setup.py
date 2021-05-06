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
    description="Example of ml project",
    author="Your name (or your organization/company/team)",
    install_requires=parse_requirements("./requirements/runtime.txt"),
    license="MIT",
)
