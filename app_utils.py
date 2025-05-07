import httpx
import threading
import logging


logger = logging.getLogger(__name__)

TIMEOUT = 600


def make_request_in_thread(url, header):
    with httpx.Client() as client:
        try:
            logger.info(f"Requesting {url}")
            response = client.get(url, timeout=TIMEOUT, headers=header)
            logger.info(f"Request completed with status: {response.status_code}")
        except Exception as e:
            logger.error(f"Request failed with error: {e}")


def wake_up_endpoints(endpoints: list[str], headers: list[dict]):
    # Start the request in a separate thread
    for endpoint, header in zip(endpoints, headers):
        request_thread = threading.Thread(target=make_request_in_thread, args=(endpoint, header))
        request_thread.daemon = True  # This makes the thread exit when the main program exits
        request_thread.start()

    logger.info("All requests started")
