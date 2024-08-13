
from django.urls import path

from user import views

urlpatterns = [
    path('login/', views.user_login, name="login"),
    path('logout/', views.user_logout, name="logout"),
    path('signup/', views.signup, name="signup"),
    path('', views.home, name="home"),
    path('participate/',views.create_participant, name="create_participant"),
    path('view/organiser/', views.view_organiser, name="view_organiser"),
    path('participate/activity/<str:id>/', views.choose_activity, name="choose_activity"),
    path('choose/activity/<str:id>/',views.choose_activity, name="choose_activity"),
]
