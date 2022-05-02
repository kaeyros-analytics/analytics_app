from django.core import serializers
from django.shortcuts import render

from webanalytics.src.dbmodels.models import webpage, pagecontent


def home(request):
    return render(request, 'home.html', {"name": "this is crazy"})



def test_db_object():
    obj1 = webpage.objects.create('testpage1', 'loclahost')
    obj1.save()
    s_obbj = serializers.serialize("page_id", [obj1])
    obj2 = pagecontent.objects.create('div_eventement', 'evenementiel', s_obbj)
    obj2.save()