import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # change to DEBUG for more detail
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

# Add handler to the logger
logger.addHandler(ch)
