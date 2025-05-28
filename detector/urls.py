from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('upload/', views.upload_image, name='upload'),
    path('result/', views.result_page, name='result'),
    path('save_result/', views.save_result_pdf, name='save_result'),
]
