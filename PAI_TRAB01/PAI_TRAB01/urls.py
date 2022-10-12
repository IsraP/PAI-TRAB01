from django.contrib import admin
from django.urls import path
from PAI.views import upload_and_crop, compare

urlpatterns = [
    path("", upload_and_crop, name="upload_and_crop"),
    path("compare", compare, name="compare")
]
