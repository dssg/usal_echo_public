from setuptools import setup, find_packages


setup(
    name='usal_echo',
    version='0.1',    
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
    entry_points='''
        [console_scripts]
        usal_echo=src.inquire:cli
    ''',
)
