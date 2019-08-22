from setuptools import setup, find_packages
import os
from pathlib import Path

conf_dir = os.path.join(str(Path.home()), "usr", "usal_echo", "conf")
os.makedirs(conf_dir, exist_ok=True)

setup(
    name="usal_echo",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    data_files=[
        (
            conf_dir,
            [
                os.path.join("conf", "examples", f)
                for f in ["path_parameters.yml", "postgres_credentials.json"]
            ]
            + [os.path.join("conf", "infra", "models_schema.sql")]
            + [os.path.join("conf", "base", "dicom_tags.json")],
        )
    ],
    entry_points="""
        [console_scripts]
        usal_echo=usal_echo.inquire:cli
    """,
)
