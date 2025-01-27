from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('convert/', views.convert_text_to_speech, name='convert_text_to_speech'),
    path('login/', views.login, name='login'),
]
