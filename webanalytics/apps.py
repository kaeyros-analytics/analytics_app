from django.apps import AppConfig
import analytics


class webAnalyticConfig(AppConfig):

    def ready(self):
        analytics.write_key = 'YOUR_WRITE_KEY'
