from django import forms
from .models import *
  
    
class PDDForm(forms.ModelForm):
#    name = forms.CharField(max_length = 100)
#    picture = forms.ImageField()
   class Meta:
        model = PddImages
        fields = ['picture']
