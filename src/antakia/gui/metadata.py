from os import path
import json
import urllib
import logging

import antakia
from importlib.resources import files

logger = logging.getLogger(__name__)


class MetaData:
    def __init__(self):
        try:
            metadata: dict = json.loads(open("counter.json", "r").read()) if path.exists("counter.json") else 0
            if isinstance(metadata, int):
                self.counter = metadata
                self.last_checked_version = None
            else:
                self.counter = metadata.get('counter')
                self.last_checked_version = metadata.get('last_checked_version')
        except Exception:
            self.counter = 0
            self.last_checked_version = None

        self.counter += 1
        logger.debug(f"GUI has been initialized {self.counter} times")
        self.latest_version = self.get_latest_version()
        self.current_version = antakia.__version__

    def get_latest_version(self):
        # Check pypi for the latest version number
        try:
            contents = urllib.request.urlopen('https://pypi.org/pypi/antakia/json').read()
            data = json.loads(contents)
            latest_version = data['info']['version']
            return latest_version
        except:
            return None

    def is_latest_version(self):
        if self.latest_version:
            return self.latest_version <= self.current_version
        return True

    def save(self):
        try:
            file = files("antakia").joinpath("assets/metadata.json")
            with open(file, "w") as f:
                f.write(json.dumps({
                    'counter': self.counter,
                    'last_checked_version': self.latest_version
                }))
        except:
            pass


metadata = MetaData()
