from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="whisper_trt",
    version="1.0",
    description="TRT implementation of Whisper Model.",
    long_description="An Optimized Speech-to-Text Pipeline for the Whisper Model.",
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=requirements,
    package_data={
        '': ['assets/*'],
    },
    include_package_data=True,
)
