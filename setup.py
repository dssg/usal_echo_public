from setuptools import setup, find_packages


setup(
    name='usal_echo',
    version='0.1',    
    package_dir={"":"src"},
    packages=find_packages(where='src'),
    include_package_data=True,
    entry_points='''
        [console_scripts]
        usal_echo=usal_echo.inquire:cli
    ''',
)
