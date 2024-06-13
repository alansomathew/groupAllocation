
from django.urls import path

from AllocationAdmin import views

urlpatterns = [
   path('index/',views.index,name="index" ), 
   path('events/',views.events,name="events" ), 
   path('event/details/<str:id>/',views.event_details,name="event_details"),
   path('event/edit/<str:id>/',views.event_edit,name="event_edit"),
   path('event/delete/<str:id>/',views.event_delete,name="event_delete"),
   path('event/participants/<str:id>/',views.list_participants,name="list_participants"),
   path('allocate_all_events/', views.allocate_all_events, name='allocate_all_events'),

]
