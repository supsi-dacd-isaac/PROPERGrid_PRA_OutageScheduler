from itertools import combinations
import logging

logger = logging.getLogger(__name__)


# core/contingency.py
def generate_n1_contingencies(n_elements, element_type="line"):
    """Returns list of N-1 contingencies.
    
    Args:
        n_elements: Number of elements to generate contingencies for
        element_type: Type of element ('line', 'gen', or 'trafo')
        
    Returns:
        List of dictionaries, each with 'element_type' and 'element_index'
    """
    return [{'element_type': element_type, 'element_index': i} for i in range(n_elements)]


def generate_nk_contingencies(n_elements, k, element_type="line"):
    """Returns list of N-k contingencies.
    
    Args:
        n_elements: Number of elements to generate contingencies for
        k: Number of simultaneous failures
        element_type: Type of element ('line', 'gen', or 'trafo')
        
    Returns:
        List of lists of dictionaries, each with 'element_type' and 'element_index'
    """
    return [[{'element_type': element_type, 'element_index': i} for i in comb] 
            for comb in combinations(range(n_elements), k)]


def apply_line_contingency(net, line_indices):
    """Switch off lines in net based on provided indices."""
    for idx in line_indices:
        net.line.loc[idx, 'in_service'] = False


def apply_nk_contingency(net, failure_event):
    """
    Apply an N-k contingency to the network by setting specified lines or generators out of service.

    Parameters:
    - net: The pandapower network object to modify
    - failure_event: A dictionary or list of dictionaries specifying the elements to take out of service.
                    Each dictionary must have:
                    'element_type' in {'line', 'gen', 'trafo'}
                    'element_index' integer
    """
    # Ensure failure_event is always treated as a list for uniformity
    if not isinstance(failure_event, list):
        failure_event = [failure_event]

    for failure_k in failure_event:
        try:
            element_type = failure_k.get('element_type')
            element_index = failure_k.get('element_index')

            if element_type == "line":
                net.line.loc[element_index, "in_service"] = False
            elif element_type == "gen":
                net.gen.loc[element_index, "in_service"] = False
            elif element_type == "trafo":
                net.trafo.loc[element_index, "in_service"] = False
            else:
                raise ValueError(f"Invalid element type for N-k....use {{line, gen, trafo}}, not {element_type}")

        except KeyError as e:
            logger.error(f"Invalid element index or type in contingency: {failure_k}. Error: {e}")
            continue  # Skip to the next failure event if an error occurs


def apply_contingency(net, cont):
    """Apply a contingency to the network.
    
    Args:
        net: The network to modify
        cont: The contingency to apply (dictionary or list of dictionaries)
    """
    if isinstance(cont, (list, tuple)):
        # For N-k contingencies
        apply_nk_contingency(net, cont)
    else:
        # For single contingencies
        apply_nk_contingency(net, [cont])


def reset_network(net):
    """Reset all lines and generators to in-service state."""
    net.line.loc[:, 'in_service'] = True
    net.gen.loc[:, 'in_service'] = True
    if hasattr(net, 'trafo'):
        net.trafo.loc[:, 'in_service'] = True


