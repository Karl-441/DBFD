from setuptools import setup, find_packages

setup(
    name="fire_correction_component",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "PyQt6>=6.6.1",
        "numpy>=1.26.4",
        "opencv-python>=4.9.0.80",
        "PyYAML>=6.0.1",
        "ultralytics>=8.1.0",
        "optuna>=3.5.0",
        "tensorboard>=2.15.1",
        "watchdog>=3.0.0",
        "Pillow>=10.2.0",
        "requests>=2.31.0",
        "pytest>=7.4.4",
        "pytest-qt>=4.4.0",
        "pytest-cov>=4.1.0",
    ],
    entry_points={
        "console_scripts": [
            "fire_correction_gui=main:main",
        ],
    },
    author="Karl_Higmut",
    description="A component for manual fire detection correction and model iteration",
    python_requires=">=3.8",
)
