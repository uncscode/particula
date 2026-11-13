# particula.logger_setup

*module*

Setup for logging in the particula package.

Based on setup from:
https://youtu.be/9L77QExPmI0?si=pSKsVyVh2dE8QxFA
https://github.com/mCodingLLC/VideosSampleCode/tree/master/videos/135_modern_logging

[View source](https://github.com/uncscode/particula/blob/main/particula/logger_setup.py#L1-L1)

## particula.logger_setup.config

*attribute* · *module attribute*

```python
config = {'version': 1, 'disable_existing_loggers': False, 'formatters': {'simple': {'format': '[%(levelname)s|%(module)s|L%(lineno)d]: %(message)s'}, 'detailed': {'format': '[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s', 'datefmt': '%Y-%m-%dT%H:%M:%S%z'}}, 'handlers': {'stderr': {'class': 'logging.StreamHandler', 'level': 'ERROR', 'formatter': 'simple', 'stream': 'ext://sys.stderr'}, 'file': {'class': 'logging.handlers.RotatingFileHandler', 'level': 'DEBUG', 'formatter': 'detailed', 'filename': os.path.join(log_dir, 'particula.log'), 'maxBytes': 25000000, 'backupCount': 3}}, 'loggers': {'particula': {'level': 'DEBUG', 'handlers': ['file', 'stderr'], 'propagate': False}, 'root': {'level': 'ERROR', 'handlers': ['stderr', 'file']}}}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/logger_setup.py#L19-L58)

## particula.logger_setup.current_dir

*attribute* · *module attribute*

```python
current_dir = os.path.dirname(os.path.abspath(__file__))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/logger_setup.py#L15-L15)

## particula.logger_setup.log_dir

*attribute* · *module attribute*

```python
log_dir = os.path.join(current_dir, 'logging')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/logger_setup.py#L17-L17)

## particula.logger_setup.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/logger_setup.py#L12-L12)

## particula.logger_setup.setup

*function*

```python
def setup()
```

Setup for logging in the particula package.

[View source](https://github.com/uncscode/particula/blob/main/particula/logger_setup.py#L61-L66)
