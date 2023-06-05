

def generate_logging_config(module_name, log_level):
    LOGGING_CONFIG = { 
            'version':1,
            'disable_existing_loggers': True,
            'formatters': { 
                'standard': { 
                    'format': f'%(levelname)s in {module_name}: %(message)s'
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
            'root': {
                'handlers': ['default'],
                'level': log_level
            }
        }
    return LOGGING_CONFIG