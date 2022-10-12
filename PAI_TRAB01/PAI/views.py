from django.shortcuts import HttpResponse, render
from django.template import loader

# Create your views here.


def upload_and_crop(request):
    template = loader.get_template('upload/upload.html')

    if request.method == "GET":
        context = {}
    elif request.method == "POST":
        context = {
            "img": request.POST.get("imgBase")
        }

    return HttpResponse(template.render(context, request))

def save_crop(request, image):
    return "ayo"

def compare(request):
    template = loader.get_template('compare/compare.html')

    return HttpResponse(template.render({}))