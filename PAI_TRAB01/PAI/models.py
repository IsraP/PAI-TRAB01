from django.db import models

# Create your models here.
class Joelho(models.Model):
    imagem = models.TextField()
    tipo = models.TextField()
    processado = models.BooleanField(default=False)
    classificado = models.BooleanField(default=False)
    rotulo = models.TextField(blank=True)
    resultadoBin = models.BooleanField(null=True)
    resultado = models.TextField()

class Resultado(models.Model):
    classificador = models.TextField()
    sensibilidade = models.FloatField()
    especificidade = models.FloatField()
    precisao = models.FloatField()
    acuracia = models.FloatField()
    escore = models.FloatField()
    tempo_treino_ms = models.FloatField()
    tempo_classificacao_ms = models.FloatField()