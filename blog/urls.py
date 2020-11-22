from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='blog-home'),
    path('about/', views.about, name='blog-about'),
    path('one/', views.one, name='blog-one'),
    path('abc/', views.abc, name='blog-abc'),
    path('one/open/', views.open, name='blog-open')
]
