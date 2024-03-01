import uuid
from os import path
import json
import urllib
import logging

import antakia
from importlib.resources import files

logger = logging.getLogger(__name__)


class MetaData:
    metadata_file = str(files("antakia").joinpath("assets/metadata.json"))

    def __init__(self):
        self.latest_version = self.get_latest_version()
        self.current_version = antakia.__version__
        self.load_file()

    def load_file(self):
        self.run_id = str(uuid.uuid4())
        try:
            metadata: dict = json.loads(open(self.metadata_file, "r").read()) if path.exists(self.metadata_file) else 0
            if isinstance(metadata, int):
                self.counter = metadata
                self.last_checked_version = None
                self.user_id = uuid.getnode()
                self.install_id = str(uuid.uuid4())
            else:
                self.counter = metadata.get('counter')
                self.last_checked_version = metadata.get('last_checked_version')
                self.user_id = metadata.get('user_id', uuid.getnode())
                self.install_id = metadata.get('install_id', str(uuid.uuid4()))
        except Exception:
            self.counter = 0
            self.last_checked_version = None
            self.user_id = uuid.getnode()
            self.install_id = str(uuid.uuid4())

    def update(self):
        self.counter += 1
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
            with open(self.metadata_file, "w") as f:
                f.write(json.dumps({
                    'counter': self.counter,
                    'last_checked_version': self.latest_version,
                    'user_id': self.user_id,
                    'install_id': self.install_id,
                }))
        except:
            pass

    def start(self):
        self.load_file()
        self.update()
        self.save()


metadata = MetaData()
