from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Event(models.Model):
    name=models.CharField(max_length=50)
    description=models.TextField(null=True,blank=True)
    min_participants=models.IntegerField()
    max_participants=models.IntegerField()
    code=models.CharField(max_length=50,unique=True)
    is_active = models.BooleanField(default=True)
    is_updated = models.BooleanField(default=False)
    is_updated_new = models.BooleanField(default=False)
    is_updated_max = models.BooleanField(default=False)
    created_by=models.ForeignKey(User,on_delete=models.CASCADE)
    created_on=models.DateTimeField(auto_now_add=True)


class Participant(models.Model):
    name=models.CharField(max_length=50)
    email=models.EmailField()
    is_active=models.BooleanField(default=True)
    created_on=models.DateTimeField(auto_now_add=True)
    updated_on=models.DateTimeField(auto_now=True)
    assigned_to=models.ForeignKey(Event,on_delete=models.CASCADE,null=True,related_name="OLD_ALLOCATION")
    assigned_to_new=models.ForeignKey(Event,on_delete=models.CASCADE,null=True,related_name='NEW_ALLOCATION')
    assigned_to_max=models.ForeignKey(Event,on_delete=models.CASCADE,null=True,related_name='MAX_ALLOCATION')


class ParticipantActivity(models.Model):
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE)
    event = models.ForeignKey(Event, on_delete=models.CASCADE)
    preference = models.IntegerField(default=0)  # Allowing negative, positive, and zero