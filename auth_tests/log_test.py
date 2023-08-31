import logging
import logging.config


LOGGING_CONFIG = { 
            'version':1,
            'disable_existing_loggers': True,
            'formatters': { 
                'standard': { 
                    'format': '%(levelname)s in reconstruct_vid: %(message)s'
                },
            },
            'handlers': {
                'default': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                }
            },
            'loggers': {
                '': {
                    'handlers': ['default'],
                    'level': 'INFO',
                    'propagate': True
                },
            },
            "root": {
                "handlers": ['default'],
                "level": "WARNING"
            }
        }
logging.config.dictConfig(LOGGING_CONFIG)
logging.info('hi')
logging.warning('hi')