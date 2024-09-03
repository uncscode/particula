"""Module for the builder classes for the general data loading settings."""

# pylint: disable=too-few-public-methods

from typing import Any, Dict, Tuple
from particula.next.abc_builder import BuilderABC
from particula.data.mixin import (
    RelativeFolderMixin,
    FilenameRegexMixin,
    FileMinSizeBytesMixin,
    HeaderRowMixin,
    DataChecksMixin,
    DataColumnMixin,
    DataHeaderMixin,
    TimeColumnMixin,
    TimeFormatMixin,
    DelimiterMixin,
    TimeShiftSecondsMixin,
    TimezoneIdentifierMixin,
    ChecksCharactersMixin,
    ChecksCharCountsMixin,
    ChecksSkipRowsMixin,
    ChecksSkipEndMixin,
    SizerConcentrationConvertFromMixin,
    SizerStartKeywordMixin,
    SizerEndKeywordMixin,
    SizerDataReaderMixin,
)


# pylint: disable=too-many-ancestors
class Loader1DSettingsBuilder(
    BuilderABC,
    RelativeFolderMixin,
    FilenameRegexMixin,
    FileMinSizeBytesMixin,
    HeaderRowMixin,
    DataChecksMixin,
    DataColumnMixin,
    DataHeaderMixin,
    TimeColumnMixin,
    TimeFormatMixin,
    DelimiterMixin,
    TimeShiftSecondsMixin,
    TimezoneIdentifierMixin,
):
    """Builder class for creating settings for loading and checking 1D data
    from CSV files."""

    def __init__(self):
        required_parameters = [
            "relative_data_folder",
            "filename_regex",
            "file_min_size_bytes",
            "header_row",
            "data_checks",
            "data_column",
            "data_header",
            "time_column",
            "time_format",
            "delimiter",
            "time_shift_seconds",
            "timezone_identifier",
        ]
        BuilderABC.__init__(self, required_parameters)
        RelativeFolderMixin.__init__(self)
        FilenameRegexMixin.__init__(self)
        FileMinSizeBytesMixin.__init__(self)
        HeaderRowMixin.__init__(self)
        DataChecksMixin.__init__(self)
        DataColumnMixin.__init__(self)
        DataHeaderMixin.__init__(self)
        TimeColumnMixin.__init__(self)
        TimeFormatMixin.__init__(self)
        DelimiterMixin.__init__(self)
        TimeShiftSecondsMixin.__init__(self)
        TimezoneIdentifierMixin.__init__(self)

    def build(self) -> Dict[str, Any]:
        """Build and return the settings dictionary for 1D data loading."""
        self.pre_build_check()
        return {
            "relative_data_folder": self.relative_data_folder,
            "filename_regex": self.filename_regex,
            "MIN_SIZE_BYTES": self.file_min_size_bytes,
            "data_loading_function": "general_1d_load",
            "header_row": self.header_row,
            "data_checks": self.data_checks,
            "data_column": self.data_column,
            "data_header": self.data_header,
            "time_column": self.time_column,
            "time_format": self.time_format,
            "delimiter": self.delimiter,
            "time_shift_seconds": self.time_shift_seconds,
            "timezone_identifier": self.timezone_identifier,
        }


class DataChecksBuilder(
    BuilderABC,
    ChecksCharactersMixin,
    ChecksCharCountsMixin,
    ChecksSkipRowsMixin,
    ChecksSkipEndMixin,
):
    """Builder class for constructing the data checks dictionary."""

    def __init__(self):
        required_parameters = [
            "characters",
            "char_counts",
            "skip_rows",
            "skip_end",
        ]
        BuilderABC.__init__(self, required_parameters)
        ChecksCharactersMixin.__init__(self)
        ChecksCharCountsMixin.__init__(self)
        ChecksSkipRowsMixin.__init__(self)
        ChecksSkipEndMixin.__init__(self)

    def build(self) -> Dict[str, Any]:
        """Build and return the data checks dictionary."""
        return {
            "characters": self.characters,
            "char_counts": self.char_counts,
            "skip_rows": self.skip_rows,
            "skip_end": self.skip_end,
        }


class SizerDataReaderBuilder(
    BuilderABC,
    SizerConcentrationConvertFromMixin,
    SizerStartKeywordMixin,
    SizerEndKeywordMixin,
):
    """Builder class for constructing the sizer data reader dictionary."""

    def __init__(self):
        required_parameters = [
            "sizer_start_keyword",
            "sizer_end_keyword",
        ]
        BuilderABC.__init__(self, required_parameters)
        SizerConcentrationConvertFromMixin.__init__(self)
        SizerStartKeywordMixin.__init__(self)
        SizerEndKeywordMixin.__init__(self)

    def build(self) -> Dict[str, Any]:
        """Build and return the sizer data reader dictionary."""
        self.pre_build_check()
        return {
            "convert_scale_from": self.sizer_concentration_convert_from,
            "Dp_start_keyword": self.sizer_start_keyword,
            "Dp_end_keyword": self.sizer_end_keyword,
        }


class LoaderSizerSettingsBuilder(
    BuilderABC,
    RelativeFolderMixin,
    FilenameRegexMixin,
    FileMinSizeBytesMixin,
    HeaderRowMixin,
    DataChecksMixin,
    DataColumnMixin,
    DataHeaderMixin,
    TimeColumnMixin,
    TimeFormatMixin,
    DelimiterMixin,
    TimeShiftSecondsMixin,
    TimezoneIdentifierMixin,
    SizerDataReaderMixin,
):
    """Builder class for creating settings for loading and checking sizer
    1D and 2D data from CSV files."""

    def __init__(self):
        required_parameters = [
            "relative_data_folder",
            "filename_regex",
            "file_min_size_bytes",
            "header_row",
            "data_checks",
            "data_column",
            "data_header",
            "time_column",
            "time_format",
            "delimiter",
            "time_shift_seconds",
            "timezone_identifier",
            "data_sizer_reader",
        ]
        BuilderABC.__init__(self, required_parameters)
        RelativeFolderMixin.__init__(self)
        FilenameRegexMixin.__init__(self)
        FileMinSizeBytesMixin.__init__(self)
        HeaderRowMixin.__init__(self)
        DataChecksMixin.__init__(self)
        DataColumnMixin.__init__(self)
        DataHeaderMixin.__init__(self)
        TimeColumnMixin.__init__(self)
        TimeFormatMixin.__init__(self)
        DelimiterMixin.__init__(self)
        TimeShiftSecondsMixin.__init__(self)
        TimezoneIdentifierMixin.__init__(self)
        SizerDataReaderMixin.__init__(self)

    def build(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build and return the two dictionaries for 1D and 2D sizer data
        loading ."""
        self.pre_build_check()
        dict_1d = {
            "relative_data_folder": self.relative_data_folder,
            "filename_regex": self.filename_regex,
            "MIN_SIZE_BYTES": self.file_min_size_bytes,
            "data_loading_function": "general_1d_load",
            "header_row": self.header_row,
            "data_checks": self.data_checks,
            "data_column": self.data_column,
            "data_header": self.data_header,
            "time_column": self.time_column,
            "time_format": self.time_format,
            "delimiter": self.delimiter,
            "time_shift_seconds": self.time_shift_seconds,
            "timezone_identifier": self.timezone_identifier,
        }
        dict_2d = {
            "relative_data_folder": self.relative_data_folder,
            "filename_regex": self.filename_regex,
            "MIN_SIZE_BYTES": self.file_min_size_bytes,
            "data_loading_function": "general_2d_load",
            "header_row": self.header_row,
            "data_checks": self.data_checks,
            "data_sizer_reader": self.data_sizer_reader,
            "data_header": self.data_header,
            "time_column": self.time_column,
            "time_format": self.time_format,
            "delimiter": self.delimiter,
            "time_shift_seconds": self.time_shift_seconds,
            "timezone_identifier": self.timezone_identifier,
        }
        return dict_1d, dict_2d
