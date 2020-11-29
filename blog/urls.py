from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='blog-home'),
    path('about/', views.about, name='blog-about'),
    path('one/', views.one, name='blog-one'),
    path('two/', views.two, name='blog-two'),
    path('three/', views.three, name='blog-three'),
    path('four/', views.four, name='blog-four'),
    path('abc/', views.abc, name='blog-abc'),
    path('one/open/', views.openpose, name='blog-open'),
    path('one/test/', views.test, name='blog-test')
]

