import base64
import os
from glob import glob
from PAI.models import Joelho
from django.db import transaction


@transaction.atomic
def saveAll(joelhos):
    for joelho in joelhos:
        joelho.save()


def populate_database(paths):
    joelhos = []

    for pathType, path in paths.items():
        filesInPath = glob(path + "/**/*.png")

        print(filesInPath)

        for file in filesInPath:
            with open(file, 'rb') as currentFile:

                label = file.split("/")
                label = label[len(label) - 2]

                encoded_string = base64.b64encode(currentFile.read()).decode("utf-8")

                newJoelho = Joelho(
                    imagem = encoded_string, 
                    tipo = pathType,
                    processado = False,
                    classificado = False,
                    resultadoBin = None,
                    resultado = "",
                    rotulo = label
                )

                joelhos.append(newJoelho)
    
    saveAll(joelhos)

def save_preprocessed(newImages):
    joelhos = []

    for img in newImages:
        newJoelho = Joelho(
            imagem = img["img"],
            tipo = img["type"],
            rotulo = img["label"],
            processado = True,
            classificado = False,
            resultadoBin = None,
            resultado = ""
        )

        joelhos.append(newJoelho)
    
    saveAll(joelhos)


        