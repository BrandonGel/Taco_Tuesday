from setuptools import setup

setup(
    name="gym_cap",
    version="0.1.4.12",
    license="MIT",
    author="DOs, skim449",
    url="https://github.com/BrandonGel/Taco_Tuesday",
    packages=["gym_cap", "gym_cap.envs", "gym_cap.heuristic"],
    zip_safe=False,
    include_package_data=True,
    install_requires = [
        "gym>=0.16.0",
        "numpy>=1.18.1"
        ]
)
