from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Any


def run_with_threads(func: Callable, inputs: List[Any], max_workers: int) -> List[Any]:
    """
    Runs a function across inputs using multithreading.

    Args:
        func (Callable): The function to run.
        inputs (List[Any]): Inputs to run the function on.
        max_workers (int): Number of threads to use.

    Returns:
        List[Any]: List of function outputs.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, inputs))
