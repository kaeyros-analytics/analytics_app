import requests


class ContactPage:
    def __init__(self, analytics):
        self.analytic = analytics

    def enter_to_page(self):
        response = self.analytic.page('2fYbTRYF69HwC2nMW9RhHp', 'contact', 'Python', {
            'url': 'http://festival.kaeyros-analytics.com/contact'
        })

        content = requests.get('https://nulldreinull-festival.de')

        return content
