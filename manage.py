#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

import analytics

from webanalytics.src.pages.contact import ContactPage


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webanalytics.settings')
    try:
        from django.core.management import execute_from_command_line

        analytics.write_key = '2fYbTRYF69HwC2nMW9RhHp'

        contact_page = ContactPage(analytics)

        response = contact_page.enter_to_page()

        print(response)

    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
