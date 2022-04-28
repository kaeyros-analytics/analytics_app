from django.db import models


class webpage(models.Model):
    page_name = models.CharField(max_length=255)
    user_ip = models.CharField(max_length=255)

    class Meta:
        app_label = 'user_app'

    def __str__(self):
        return self.page_name  # "%s %s" % (self.page_name, self.user_ip)
