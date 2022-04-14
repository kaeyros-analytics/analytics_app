from django.http import HttpRequest, JsonResponse
from rest_framework.decorators import api_view


@api_view(['GET'])
def ip(request: HttpRequest):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip_add = x_forwarded_for.split(',')[0]
    else:
        ip_add = request.META.get('REMOTE_ADDR')

    return JsonResponse(data={'ip': ip_add})

@api_view(['POST'])
def save_link(req: HttpRequest):
    links = req.data
    print(links)
    # TODO

    return JsonResponse({'value': True})
