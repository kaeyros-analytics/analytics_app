from django.db import models

from webanalytics.src.models import webpage


class pagecontent(models.Model):
    pagename = models.ForeignKey(webpage, on_delete=models.CASCADE())
    content_tag = models.CharField(max_length=255)
    content_title = models.CharField(max_length=255)

    def __str__(self):
        return self.content_title
