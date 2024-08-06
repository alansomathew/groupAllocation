
from django.urls import path

from AllocationAdmin import views

urlpatterns = [
    path('index/', views.index, name="index"),
    path('events/', views.events, name="events"),
    path('event/details/<str:id>/', views.event_details, name="event_details"),
    path('event/edit/<str:id>/', views.event_edit, name="event_edit"),
    path('event/delete/<str:id>/', views.event_delete, name="event_delete"),
    path('event/participants/<str:id>/',views.list_participants, name="list_participants"),
    path('allocate/', views.allocate_participants, name='allocate_participants'),
    path('view-allocation/', views.view_allocation, name='view_allocation'),
    path('allocate-new/', views.allocate_participants_new,name='allocate_participants_new'),
    path('view-allocation-new/', views.view_allocation_new,name='view_allocation_new'),

]
