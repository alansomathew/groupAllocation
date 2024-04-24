from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Event(models.Model):
    name=models.CharField(max_length=50)
    description=models.TextField()
    min_participants=models.IntegerField()
    max_participants=models.IntegerField()
    code=models.CharField(max_length=50,unique=True)
    is_active = models.BooleanField(default=True)
    created_by=models.ForeignKey(User,on_delete=models.CASCADE)
    created_on=models.DateTimeField(auto_now_add=True)


class Participant(models.Model):
    name=models.CharField(max_length=50)
    email=models.EmailField()
    is_active=models.BooleanField(default=True)
    created_on=models.DateTimeField(auto_now_add=True)
    updated_on=models.DateTimeField(auto_now=True)

class ParticipantActivity(models.Model):
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE)
    activity = models.ForeignKey(Event, on_delete=models.CASCADE)