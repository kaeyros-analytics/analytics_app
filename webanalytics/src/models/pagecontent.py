from django.db import models

from webanalytics.src.models import webpage


class pagecontent(models.Model):
    content_tag = models.CharField(max_length=255)
    content_title = models.CharField(max_length=255)

    def __init__(self):
        self.pagename = models.ForeignKey(self, on_delete=models.CASCADE)

    def __str__(self):
        return self.content_title
