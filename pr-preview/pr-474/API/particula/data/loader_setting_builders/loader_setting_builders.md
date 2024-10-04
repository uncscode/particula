# Loader Setting Builders

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Data](./index.md#data) / Loader Setting Builders

> Auto-generated documentation for [particula.data.loader_setting_builders](https://github.com/uncscode/particula/blob/main/particula/data/loader_setting_builders.py) module.

## DataChecksBuilder

[Show source in loader_setting_builders.py:100](https://github.com/uncscode/particula/blob/main/particula/data/loader_setting_builders.py#L100)

Builder class for constructing the data checks dictionary.

#### Signature

```python
class DataChecksBuilder(
    BuilderABC,
    ChecksCharactersMixin,
    ChecksCharCountsMixin,
    ChecksReplaceCharsMixin,
    ChecksSkipRowsMixin,
    ChecksSkipEndMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../next/abc_builder.md#builderabc)
- [ChecksCharCountsMixin](./mixin.md#checkscharcountsmixin)
- [ChecksCharactersMixin](./mixin.md#checkscharactersmixin)
- [ChecksReplaceCharsMixin](./mixin.md#checksreplacecharsmixin)
- [ChecksSkipEndMixin](./mixin.md#checksskipendmixin)
- [ChecksSkipRowsMixin](./mixin.md#checksskiprowsmixin)

### DataChecksBuilder().build

[Show source in loader_setting_builders.py:125](https://github.com/uncscode/particula/blob/main/particula/data/loader_setting_builders.py#L125)

Build and return the data checks dictionary.

#### Signature

```python
def build(self) -> Dict[str, Any]: ...
```



## Loader1DSettingsBuilder

[Show source in loader_setting_builders.py:33](https://github.com/uncscode/particula/blob/main/particula/data/loader_setting_builders.py#L33)

Builder class for creating settings for loading and checking 1D data
from CSV files.

#### Signature

```python
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
    def __init__(self): ...
```

#### See also

- [BuilderABC](../next/abc_builder.md#builderabc)
- [DataChecksMixin](./mixin.md#datachecksmixin)
- [DataColumnMixin](./mixin.md#datacolumnmixin)
- [DataHeaderMixin](./mixin.md#dataheadermixin)
- [DelimiterMixin](./mixin.md#delimitermixin)
- [FileMinSizeBytesMixin](./mixin.md#fileminsizebytesmixin)
- [FilenameRegexMixin](./mixin.md#filenameregexmixin)
- [HeaderRowMixin](./mixin.md#headerrowmixin)
- [RelativeFolderMixin](./mixin.md#relativefoldermixin)
- [TimeColumnMixin](./mixin.md#timecolumnmixin)
- [TimeFormatMixin](./mixin.md#timeformatmixin)
- [TimeShiftSecondsMixin](./mixin.md#timeshiftsecondsmixin)
- [TimezoneIdentifierMixin](./mixin.md#timezoneidentifiermixin)

### Loader1DSettingsBuilder().build

[Show source in loader_setting_builders.py:80](https://github.com/uncscode/particula/blob/main/particula/data/loader_setting_builders.py#L80)

Build and return the settings dictionary for 1D data loading.

#### Signature

```python
def build(self) -> Dict[str, Any]: ...
```



## LoaderSizerSettingsBuilder

[Show source in loader_setting_builders.py:164](https://github.com/uncscode/particula/blob/main/particula/data/loader_setting_builders.py#L164)

Builder class for creating settings for loading and checking sizer
1D and 2D data from CSV files.

#### Signature

```python
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
    def __init__(self): ...
```

#### See also

- [BuilderABC](../next/abc_builder.md#builderabc)
- [DataChecksMixin](./mixin.md#datachecksmixin)
- [DataColumnMixin](./mixin.md#datacolumnmixin)
- [DataHeaderMixin](./mixin.md#dataheadermixin)
- [DelimiterMixin](./mixin.md#delimitermixin)
- [FileMinSizeBytesMixin](./mixin.md#fileminsizebytesmixin)
- [FilenameRegexMixin](./mixin.md#filenameregexmixin)
- [HeaderRowMixin](./mixin.md#headerrowmixin)
- [RelativeFolderMixin](./mixin.md#relativefoldermixin)
- [SizerDataReaderMixin](./mixin.md#sizerdatareadermixin)
- [TimeColumnMixin](./mixin.md#timecolumnmixin)
- [TimeFormatMixin](./mixin.md#timeformatmixin)
- [TimeShiftSecondsMixin](./mixin.md#timeshiftsecondsmixin)
- [TimezoneIdentifierMixin](./mixin.md#timezoneidentifiermixin)

### LoaderSizerSettingsBuilder().build

[Show source in loader_setting_builders.py:214](https://github.com/uncscode/particula/blob/main/particula/data/loader_setting_builders.py#L214)

Build and return the two dictionaries for 1D and 2D sizer data
loading .

#### Signature

```python
def build(self) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...
```



## SizerDataReaderBuilder

[Show source in loader_setting_builders.py:136](https://github.com/uncscode/particula/blob/main/particula/data/loader_setting_builders.py#L136)

Builder class for constructing the sizer data reader dictionary.

#### Signature

```python
class SizerDataReaderBuilder(
    BuilderABC,
    SizerConcentrationConvertFromMixin,
    SizerStartKeywordMixin,
    SizerEndKeywordMixin,
):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../next/abc_builder.md#builderabc)
- [SizerConcentrationConvertFromMixin](./mixin.md#sizerconcentrationconvertfrommixin)
- [SizerEndKeywordMixin](./mixin.md#sizerendkeywordmixin)
- [SizerStartKeywordMixin](./mixin.md#sizerstartkeywordmixin)

### SizerDataReaderBuilder().build

[Show source in loader_setting_builders.py:154](https://github.com/uncscode/particula/blob/main/particula/data/loader_setting_builders.py#L154)

Build and return the sizer data reader dictionary.

#### Signature

```python
def build(self) -> Dict[str, Any]: ...
```
