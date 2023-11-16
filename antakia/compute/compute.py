# umpa fires NumbaDeprecationWarning warning, so :
import warnings
from numba.core.errors import NumbaDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=UserWarning)

from antakia.utils import conf_logger
import logging

logger = logging.getLogger(__name__)
conf_logger(logger)






