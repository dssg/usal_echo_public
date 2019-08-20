from setuptools import setup, find_packages


setup(
    name='cibercv-usal',
    version='0.1',
    packages=find_packages(),
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
    py_modules = ['src.d01_data.ingestion_dcm', 'src.d01_data.ingestion_xtdb', 
                    'src.d02_intermediate.clean_dcm', 'src.d02_intermediate.clean_xtdb'],
    entry_points='''
        [console_scripts]
        usal=src.inquire:cli
    ''',
)