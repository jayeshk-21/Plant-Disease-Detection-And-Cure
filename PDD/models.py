from django.db import models

# Create your models here.

class PddImages(models.Model):
    picture = models.ImageField(upload_to = 'images/')
    disease_type = models.CharField(max_length=500, null=True)
    date_uploaded=models.DateTimeField(auto_now_add=True, null=True)
    pesticides = models.CharField(max_length=500, null=True)