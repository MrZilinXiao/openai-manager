import setuptools

with open("README.md", "r") as fh:
    _LONG_DESCRIPTION = fh.read()


setuptools.setup(
    name="openai-manager",
    license='MIT',
    author="MrZilinXiao",
    version="0.0.3",
    author_email="me@mrxiao.net",
    description="Speed up your OpenAI requests by balancing prompts to multiple API keys.",
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/MrZilinXiao/openai-manager",
    packages=setuptools.find_namespace_packages(),
    install_requires=["aiohttp", "pytest", "openai", "tiktoken"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7'
)
