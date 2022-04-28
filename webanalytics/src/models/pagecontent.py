from django.db import models

from webanalytics.src.models import webpage


class pagecontent(models.Model):
    web_page = models.ForeignKey(webpage.webpage, on_delete=models.CASCADE)
    content_tag = models.CharField(max_length=255)
    content_title = models.CharField(max_length=255)

    class Meta:
        app_label = 'user_app'

    def __str__(self):
        return self.web_page.page_name
