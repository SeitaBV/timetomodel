from setuptools import setup

setup(
    name="timetomodel",
    description="Sane handling of time series data for forecast modelling - with production usage in mind.",
    author="Seita BV",
    author_email="nicolas@seita.nl",
    keywords=["time series", "forecasting"],
    version="0.6.1",
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
    packages=["timetomodel", "timetomodel.utils"],
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
    Sane handling of time series data for forecast modelling - with production usage in mind.
    While modelling time series data with data science libraries like Pandas, statsmodels, sklearn etc.,
    dealing with time series data is cumbersome - timetomodel takes some of that over. Loading data, making
    train/test data, feeding data into rolling forecasts...
    Also, the context and assumptions under which a model was made and used should not be in notebooks, they should
    have a readable and reproducible spec.
    Timetomodel is hopefully useful while doing data & model exploration as well as when integrating or replacing 
    models in production environments.
    """,
)
