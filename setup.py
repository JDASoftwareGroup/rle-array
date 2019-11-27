import os
from typing import List

from setuptools import setup


def get_requirements(path: str) -> List[str]:
    with open(os.path.join(os.path.dirname(__file__), path)) as fp:
        content = fp.read()
    return [req for req in content.split("\n") if req != "" and not req.startswith("#")]


def setup_package() -> None:
    name = "rle_array"

    setup(
        name=name,
        packages=["rle_array"],
        description="Run-length encoded pandas.",
        author="JDA Software, Inc",
        python_requires=">=3.6",
        url="https://github.com/JDASoftwareGroup/rle_array",
        license="MIT",
        long_description=open("README.rst", "r").read(),
        install_requires=get_requirements("requirements.txt"),
        tests_require=get_requirements("test-requirements.txt"),
        extras_require={"testing": get_requirements("test-requirements.txt")},
        keywords=["python"],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Environment :: Console",
            "Intended Audience :: Developers",
            "Natural Language :: English",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
        ],
        use_scm_version=True,
        command_options={
            "build_sphinx": {
                "source_dir": ("setup.py", "docs"),
                "build_dir": ("setup.py", "docs/_build"),
                "builder": ("setup.py", "doctest,html"),
                "warning_is_error": ("setup.py", "1"),
            }
        },
    )


if __name__ == "__main__":
    setup_package()
