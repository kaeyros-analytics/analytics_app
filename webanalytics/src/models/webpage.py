from django.db import models


class webpage(models.Model):
    page_name = models.CharField(max_length=255)
    user_ip = models.CharField(max_length=255)

    def __str__(self):
        return "%s %s" % (self.page_name, self.user_ip)
