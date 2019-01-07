from setuptools import setup

setup(
    name="ts_forecasting_pipeline",
    description="Toolset for time series forecasting. Supports both basic modeling and in-production usage.",
    author="Seita BV",
    author_email="nicolas@seita.nl",
    keywords=["time series", "forecasting"],
    version="0.4.0",
    install_requires=[
        "pandas",
        "statsmodels",
        "sklearn",
        "matplotlib",
        "numpy",
        "scipy",
        "pytz",
        "python-dateutil >= 2.5",
        "SQLAlchemy",
    ],
    tests_require=["pytest"],
    packages=["ts_forecasting_pipeline", "ts_forecasting_pipeline.utils"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    long_description="""\
    Toolset for time series forecasting, based on fundamental data science libraries like Pandas,
    statsmodels, sklearn etc.
    Supports model design as well as storing and accessing forecasts.
    It contains functionality that is both useful for data & model exploration as well as 
    integrating into production code.
    """,
)
