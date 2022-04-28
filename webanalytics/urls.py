"""webanalytics URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

import logging
import analytics

from webanalytics import views
from webanalytics.src.controller import home_controller

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/ip', home_controller.ip),
    path('api/link-data', home_controller.save_link),
    path('home', views.home)
]

analytics.debug = True
analytics.write_key = 'YOUR_WRITE_KEY'
