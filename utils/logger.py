import logging
import os
import sys


class Logger(object):

    flag_handlers = False

    """Sets handlers according to the calling module and log file path"""
    @classmethod
    def set_handlers(cls, name, path, level='debug'):
        cls.logger = logging.getLogger(name)
        cls.logger.setLevel(logging.DEBUG)

        # create a file handler (level = DEBUG)
        cls.file_handler = logging.FileHandler(path)
        cls.file_handler.setLevel(logging.DEBUG)

        # create a stdout handler
        cls.stdout_handler = logging.StreamHandler(sys.stdout)
        if level == 'debug':
            cls.stdout_handler.setLevel(logging.DEBUG)
        elif level == 'info':
            cls.stdout_handler.setLevel(logging.INFO)

        # create a logging format
        cls.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        cls.file_handler.setFormatter(cls.formatter)
        cls.stdout_handler.setFormatter(cls.formatter)

        # add the handlers to the logger
        cls.logger.addHandler(cls.file_handler)
        cls.logger.addHandler(cls.stdout_handler)

        cls.flag_handlers = True

    """Logs INFO level"""
    @classmethod
    def info(cls, content):
        if not cls.flag_handlers:
            assert False, "must set handlers using set_handlers() before logging anything"

        Logger.logger.info(content)

    """ Logs DEBUG level"""
    @classmethod
    def debug(cls, content):
        if not cls.flag_handlers:
            assert False, "must set handlers using set_handlers() before logging anything"
        Logger.logger.debug(content)

    """ Logs WARNING level"""
    @classmethod
    def warning(cls, content):
        if not cls.flag_handlers:
            assert False, "must set handlers using set_handlers() before logging anything"
        Logger.logger.warning(content)


def init_debug_log(time, log_dir, file_prefix, verbose_flag=True):
    """
    create log object
    :param time: time to insert into log file title
    :param log_dir: path to log dir
    :param file_prefix: file name
    :param verbose_flag: whether to print logs in addition to save in log file or not.
    :return: logging object
    """
    log_file_name = log_dir + file_prefix + '_' + str(time) + '.log'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(filename=log_file_name,
                        format='%(asctime)s, %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    # print result in addition to log file if verbose flag is true
    if verbose_flag:
        stderrLogger = logging.StreamHandler()
        stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        logging.getLogger().addHandler(stderrLogger)

    logging.info("start log program")
    return logging

