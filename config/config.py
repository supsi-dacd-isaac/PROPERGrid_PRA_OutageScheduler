from dataclasses import dataclass
from utils.utils import *

""" configurations blueprint"""
@dataclass(frozen=True)  # Instances of this class are immutable.
class ConfigData:
    add_variables_msg = (f'{blue_c}Variables Added:{reset_c}\n'
                         f' - {blue_c}xt, sxt, ext{reset_c}: s Scheduled outage decisions, start-end indicators for the outage task\n'
                         f' - {blue_c}pgen, pgen_c{reset_c}: Generated power planned and N-1 states\n'
                         f' - {blue_c}d_cut, d_cut_c{reset_c}: loss of load planned and N-1 states\n'
                         f' - {blue_c}f, f_c{reset_c}: Power flow in planned and N-1 states\n')

    objective_html = r""" 
                          <div style=\"color:orange;\"> 
                          \[ \max_X \Bigl( \sum_o C_{PM,o}(X) - \omega_1 \sum_t\sum_b C_{VoLL,t,b}(X) \\
                                 - \omega_2 \sum_t\sum_b\sum_C C_{VoLL,t,b}(X|C)    
                                 - \omega_3 \sum_t\sum_b C_{OP,b,t}(X)    
                                 - \omega_4 \sum_t\sum_b\sum_C C_{OP,b,t}(X|C) \Bigr) \] </div> 
                      """



def set_logger_config():
    """     Configures the logger for the application. """
    # Create a logger
    logger = logging.getLogger('OutageScheduler')
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('outage_scheduler.log')

    # Set level of handlers
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.ERROR)

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger