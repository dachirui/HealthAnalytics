from django.conf.urls import url
from django.contrib import admin
from django.urls import path

from TweetApp import views
from templates import *

urlpatterns = [
    url(r'^grafico/static/',views.getJson),
    url(r'^final/static/',views.getJson),
    url(r'^signup/$', views.signupform),
    url(r'^login/$', views.login),
    url(r'^login/result.html$', views.result),
    url(r'^login/config.html$', views.config),
    url(r'^login/registro.html$', views.mostrarRegistro),
    url(r'^final/$', views.graficos),
]