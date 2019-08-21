from setuptools import setup, find_packages
import os
from pathlib import Path

usr_dir = os.path.join(str(Path.home()),'usr','usal_echo','conf')
os.makedirs(usr_dir, exist_ok=True)

setup(
    name='usal_echo',
    version='0.1',    
    package_dir={"":"src"},
    packages=find_packages(where='src'),
    include_package_data=True,
    install_requires=[
        'pandas==0.25.0',
        'sqlalchemy==1.3.6',
        'pyyaml==5.1',
        'pydicom==1.3.0',
        'shapely==1.6.4',
        'click==7.0',
        'PyInquirer==1.0.3',
    ],
    data_files=[(usr_dir, [os.path.join('conf','local', f) for f in ['path_parameters.yml','postgres_credentials.json']])],
    entry_points='''
        [console_scripts]
        usal_echo=usal_echo.inquire:cli
    ''',
)
