import json
import os
import random
from importlib.resources import files

import requests

from antakia import config
from antakia.gui.metadata import metadata


class ActivityLogger:
    log_file = str(files("antakia").joinpath("assets/logs.json"))
    url = 'https://api.antakia.ai/'
    limit = 1
    send_events = ['launched']

    def __init__(self):
        self.logs = []
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as log_file:
                for log in log_file:
                    self.logs.append(json.loads(log))

    def log(self, event: str, info: dict | None = None):
        """
        logs the user activity for usage statistics

        Parameters
        ----------
        event: user action
        info: mate information

        Returns
        -------

        """
        if not os.environ.get('ATK_SEND_LOG', 1) or not config.ATK_SEND_LOG:
            return
        if info is None:
            info = {}
        payload = {
            'event': event,
            'log': info
        }
        try:
            json.dumps(payload)
            self.add_to_log_queue(payload)
        except:
            pass

    def add_metadata(self, payload):
        payload['user_id'] = metadata.user_id
        payload['run_id'] = metadata.run_id
        payload['install_id'] = metadata.install_id
        payload['launch_count'] = metadata.counter
        payload['version_number'] = metadata.current_version

    def add_to_log_queue(self, payload):
        self.logs.append(payload)
        self.add_to_disk(payload)
        if len(self.logs) > self.limit or payload['event'] in self.send_events:
            self.send()

    def add_to_disk(self, payload):
        try:
            with open(self.log_file, 'a') as log_file:
                log_file.write(json.dumps(payload) + '\n')
        except:
            pass

    def clear_logs(self):
        self.logs = []
        try:
            with open(self.log_file, 'w') as log_file:
                log_file.write('')
        except:
            pass

    def send(self):
        print('try send')
        try:
            payload = {'items': self.logs}
            self.add_metadata(payload)
            print(payload)
            response = requests.post(self.url + 'log/', json=payload)
            print(response)
            if response.status_code < 300 or random.random() > 0.9:
                self.clear_logs()
                print('log sent')
        except:
            pass


stats_logger = ActivityLogger()
