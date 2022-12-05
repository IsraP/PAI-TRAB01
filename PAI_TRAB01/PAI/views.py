from django.shortcuts import HttpResponse, render, redirect
from django.template import loader
from PAI.service.image_service import findCropInImage
from PAI.service.image_service import preprocess as preprocessImage
from PAI.service.persistence_service import populate_database, saveAll
from PAI.service.knn_service import knn_classify, knn_classify_binary
from PAI.models import Joelho

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

def manipulate(request):
    template = loader.get_template('manipulate/manipulate.html')

    context = {}

    if request.method == "POST":
        trainPath = request.POST.get("trainPath")
        validatePath = request.POST.get("validatePath")
        testPath = request.POST.get("testPath")

        paths = {
            "train": trainPath, 
            "validate": validatePath, 
            "test": testPath}

        populate_database(paths)


    return HttpResponse(template.render(context, request))

def preprocess(request):
    joelhos = Joelho.objects.filter(processado=False)

    preprocessImage(joelhos)

    for joelho in joelhos:
        joelho.processado = True

    saveAll(joelhos)    

    return redirect(manipulate)

def classify(request):
    template = loader.get_template('classify/classify.html')

    context = {}

    return HttpResponse(template.render(context, request))

def classify_knn(request):
    joelhosTreino = Joelho.objects.filter(processado=True, classificado=False, tipo='train')
    joelhosTeste = Joelho.objects.filter(processado=True, classificado=False, tipo='test')
    joelhosValid = Joelho.objects.filter(processado=True, classificado=False, tipo='validate')

    result, matrix = knn_classify(joelhosTreino, joelhosTeste, joelhosValid)

    result.save()

    return redirect(manipulate)

def classify_knn_binary(request):
    joelhosTreino = Joelho.objects.filter(processado=True, classificado=False, tipo='train')
    joelhosTeste = Joelho.objects.filter(processado=True, classificado=False, tipo='test')
    joelhosValid = Joelho.objects.filter(processado=True, classificado=False, tipo='validate')

    result, matrix = knn_classify_binary(joelhosTreino, joelhosTeste, joelhosValid)

    result.save()

    return redirect(classify)