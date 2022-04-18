import json

from django.http import HttpRequest, JsonResponse
from rest_framework.decorators import api_view

from webanalytics.src.models.pagecontent import pagecontent
from webanalytics.src.models.webpage import webpage


@api_view(['GET'])
def ip(request: HttpRequest):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip_add = x_forwarded_for.split(',')[0]
    else:
        ip_add = request.META.get('REMOTE_ADDR')

    print(ip_add)
    return JsonResponse(data={'ip': ip_add})


@api_view(['POST'])
def save_link(req: HttpRequest):
    links = req.data
    print(links)
    # TODO
    seite = webpage(page_name=req.META.get('REMOTE_ADDR'), user_ip=req.META.get('HTTP_X_FORWARDED_FOR'))
    seite.save()

    obj = json.loads(links)
    title = obj['title']
    contentag = obj['content']
    content = pagecontent(pagename=seite, content_title=title, content_tag=contentag)
    content.save()

    return JsonResponse({'value': True})
