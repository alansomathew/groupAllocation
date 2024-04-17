
from django.urls import path

from AllocationAdmin import views

urlpatterns = [
   path('index/',views.index,name="index" ),  
]
