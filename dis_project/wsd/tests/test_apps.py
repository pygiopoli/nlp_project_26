import os
from unittest.mock import patch

from django.apps import apps
from django.test import SimpleTestCase


class WarmUpOnStartTests(SimpleTestCase):
    def test_warms_up_in_the_serving_process(self) -> None:
        with patch.dict(os.environ, {"RUN_MAIN": "true"}), patch("wsd.apps.warm_up") as warm_up:
            apps.get_app_config("wsd").ready()
        warm_up.assert_called_once()

    def test_does_not_warm_up_in_other_processes(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("wsd.apps.warm_up") as warm_up:
            apps.get_app_config("wsd").ready()
        warm_up.assert_not_called()
