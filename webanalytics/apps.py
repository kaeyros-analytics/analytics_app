from django.apps import AppConfig
import analytics
from src.pages.contact import ContactPage


class WebAnalyticConfig(AppConfig):
    def ready(self):
        analytics.write_key = 'YOUR_WRITE_KEY'

        contact_page = ContactPage(analytics)

        response = contact_page.enter_to_page()

        print(response)
