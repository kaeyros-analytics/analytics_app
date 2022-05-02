from rest_framework import serializers

from webanalytics.src.dbmodels.models import pagecontent, webpage


class WebPageSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = webpage
        fields = ('id', 'page_name', 'user_ip')


class PageContentSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = pagecontent
        fields = ('id', 'content_tag', 'content_title', 'page_id')
