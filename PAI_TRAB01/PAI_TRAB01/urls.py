from django.contrib import admin
from django.urls import path
from PAI.views import upload_and_crop, compare, manipulate, preprocess, classify, classify_knn, classify_knn_binary

urlpatterns = [
    path("", upload_and_crop, name="upload_and_crop"),
    path("compare", compare, name="compare"),
    path("manipulate", manipulate, name="manipulate"),
    path("preprocess", preprocess, name="preprocess"),
    path("classify", classify, name="classify"),
    path("classify-knn", classify_knn, name="classify-knn"),
    path("classify-knn-binary", classify_knn_binary, name="classify-knn-binary")
]
