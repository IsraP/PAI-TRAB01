from django.shortcuts import HttpResponse, render
from django.template import loader
from PAI.service.image_service import findCropInImage

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


def compare(request):
    template = loader.get_template('compare/compare.html')

    if request.method == "GET":
        context = {}
    elif request.method == "POST":
        templateImg = request.POST.get("template")
        img = request.POST.get("img")
        result = findCropInImage(templateImg, img)

        context = {
            "template": templateImg,
            "img": img,
            "compare": result,
            "error": result == None
        }

    return HttpResponse(template.render(context, request))