"""Exceptions that are worth catching"""


class IncompatibleModelSpecs(Exception):
    """Model specs will lead to problems when used."""

    pass


class MissingData(Exception):
    """Data is missing where there needs to be some."""

    pass


class NaNData(Exception):
    """Data is NaN where there needs to be some."""

    pass


class UnsupportedModel(Exception):
    """If a statistic model is not implemented in this tool (yet)."""

    pass
