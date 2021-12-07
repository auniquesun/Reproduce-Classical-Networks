import logging


class Logger(object):
    def __init__(self, args):
        logger = logging.getLogger(__name__)

        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(args.log_file)
        formater = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        file_handler.setFormatter(formater)

        logger.addHandler(file_handler)

        return logger
