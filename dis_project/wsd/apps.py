import os

from django.apps import AppConfig

from algorithms.nltk_utils import warm_up


class WsdConfig(AppConfig):
    name = 'wsd'

    def ready(self) -> None:
        # `runserver` starts two processes: one watches the files for changes,
        # the other serves the requests. Django sets RUN_MAIN="true" only in
        # the second one, so the resources are loaded once, where they are
        # used, and not during other commands such as `test` or `check`.
        if os.environ.get("RUN_MAIN") == "true":
            warm_up()
