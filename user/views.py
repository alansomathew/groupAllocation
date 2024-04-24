from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.urls import reverse
from AllocationAdmin.models import Event, Participant, ParticipantActivity


# Create your views here.


def user_login(request):
    try:
        if request.method == 'POST':
            username = request.POST.get('email').strip()
            password = request.POST.get('pass').strip()
            user = authenticate(
                request, username=username, password=password)
            if user is not None:
                # print(user)
                login(request, user)
                if user.is_superuser:
                    login(request, user)
                    request.session['user_id'] = user.id
                    # print(user)
                    return redirect("index")

                else:
                    login(request, user)
                    # Set the user ID in the session
                    request.session['user_id'] = user.id
                    return redirect("index")
            else:
                messages.error(request, 'Invalid email or password')
                return render(request, 'Guest/Login.html')
        else:
            return render(request, 'Guest/Login.html')
    except Exception as e:
        print(e)
        messages.error(request, 'Error viewing data!')
        return render(request, 'Guest/Login.html')


def user_logout(request):
    logout(request)
    return redirect('home')


def signup(request):
    try:
        if request.method == 'POST':
            password = request.POST.get('txtPass').strip()
            first_name = request.POST.get('txtFname').strip()
            last_name = request.POST.get('txtLname').strip()
            email = request.POST.get('txtEmail').strip()

            user = User.objects.create_user(
                username=email, password=password, first_name=first_name, last_name=last_name,  email=email)
            user.save()
            messages.success(request, 'Account created successfully')
            return redirect('login')
        else:
            return render(request, 'Guest/Signup.html')
    except Exception as e:
        print(e)
        messages.error(request, 'Error viewing data!')
        return render(request, 'Guest/Signup.html')


def home(request):
    return render(request, 'Guest/Home.html')

def create_participant(request):
    if request.method == 'POST':
        # Extract data from POST request
        
        participant_name = request.POST.get('txtn')
        participant_email = request.POST.get('email')

        # Create and save Participant instance
        participant = Participant(
           
            name=participant_name,
            email=participant_email,
        )
        participant.save()
        return redirect('choose_activity',id=participant.id)

    return render(request, 'Guest/Event.html')

def choose_activity(request,id):
    participantObj=Participant.objects.get(id=id)
    data = Event.objects.filter(is_active=True)
    if request.method == 'POST':
        selected_activities = request.POST.getlist('activities')
        
        # Iterate over selected activities and save them for the participant
        for activity_id in selected_activities:
            activity = Event.objects.get(pk=activity_id)
            ParticipantActivity.objects.create(participant=participantObj, activity=activity)
            
        return redirect('home')
    else:
        return render(request, 'Guest/activity.html', {'data': data, 'participant':participantObj})
