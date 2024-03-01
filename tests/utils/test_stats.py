import os
from unittest import TestCase
import mock
import pandas as pd
import pytest

from antakia import config
from antakia.utils import stats


class DummyResponse:
    def __init__(self, status_code=None):
        if status_code is None:
            self.status_code = 200
        else:
            self.status_code = status_code


def check_log_file():
    with open(stats.stats_logger.log_file, 'r') as f:
        num_lines = sum(1 for _ in f)
    assert num_lines == len(stats.stats_logger._logs)


class TestStats(TestCase):
    def setUp(self):
        stats.stats_logger = stats.ActivityLogger('test.json')
        stats.stats_logger._clear_logs()

    def tearDown(self):
        file = stats.stats_logger.log_file
        if os.path.isfile(file):
            os.remove(file)

    def test_log(self):
        assert len(stats.stats_logger._logs) == 0
        # add log with no info
        stats.stats_logger.log('test1')
        assert len(stats.stats_logger._logs) == 1
        check_log_file()

        # add log with info
        stats.stats_logger.log('test1', {'d': 2})
        assert len(stats.stats_logger._logs) == 2
        check_log_file()

        # add log with non json dtype -> ignored
        stats.stats_logger.log('test1', {'d': pd.DataFrame()})
        assert len(stats.stats_logger._logs) == 2
        check_log_file()

    @mock.patch('antakia.utils.stats.requests.post')
    def test_send(self, post):
        post.return_value = DummyResponse()
        assert len(stats.stats_logger._logs) == 0
        check_log_file()

        # send 1 log
        stats.stats_logger.limit = 0
        stats.stats_logger.log('test1', {'item1': 1})
        assert post.call_count == 1
        assert len(stats.stats_logger._logs) == 0
        check_log_file()
        payload = post.call_args[1]['json']
        for k in ['items', 'user_id', 'run_id', 'install_id', 'launch_count', 'version_number']:
            assert k in payload
        assert len(payload['items']) == 1
        assert 'item1' in payload['items'][0]['log']
        assert 'test1' == payload['items'][0]['event']

        # fail first sent, then succeed
        post.return_value = DummyResponse(300)
        stats.stats_logger.log('test1', {'item1': 1})
        assert post.call_count == 2
        assert len(stats.stats_logger._logs) == 1
        check_log_file()
        post.return_value = DummyResponse(200)
        stats.stats_logger.log('test1', {'item1': 1})
        assert post.call_count == 3
        assert len(post.call_args[1]['json']['items']) == 2
        assert len(stats.stats_logger._logs) == 0

        # fail send and exceed size limit
        post.return_value = DummyResponse(300)
        stats.stats_logger.size_limit = 0
        stats.stats_logger.log('test1', {'item1': 1})
        assert post.call_count == 4
        assert len(stats.stats_logger._logs) == 0
        check_log_file()

        # test send prio event
        post.return_value = DummyResponse(200)
        stats.stats_logger.size_limit = 10
        stats.stats_logger.limit = 10
        stats.stats_logger.log(stats.stats_logger.send_events[1], {'item1': 1})
        assert post.call_count == 5
        assert len(stats.stats_logger._logs) == 0
        check_log_file()

    @mock.patch('antakia.utils.stats.requests.post')
    def test_log_error(self, post):
        @stats.log_errors
        def test_fct():
            raise KeyError

        post.return_value = DummyResponse(200)

        with pytest.raises(KeyError):
            test_fct()
        assert post.call_count == 1
        assert len(post.call_args[1]['json']['items']) == 1
        assert len(stats.stats_logger._logs) == 0
        check_log_file()

    def test_no_log(self):
        assert len(stats.stats_logger._logs) == 0

        config.ATK_SEND_LOG = False

        stats.stats_logger.log('test1')
        assert len(stats.stats_logger._logs) == 0
        check_log_file()

        config.ATK_SEND_LOG = True
        os.environ['ATK_SEND_LOG'] = '0'

        stats.stats_logger.log('test1')
        assert len(stats.stats_logger._logs) == 0
        check_log_file()

        os.environ['ATK_SEND_LOG'] = '1'
