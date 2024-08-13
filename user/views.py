from django.shortcuts import get_object_or_404, render, redirect
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


def view_organiser(request):
    try:
        # Get distinct users who have created events
        organisers = User.objects.filter(event__isnull=False).distinct()

        # Render the result to the template
        return render(request, 'Guest/view_organiser.html',{"data":organisers})
    except Exception as e:
        print(e)
        messages.error(request, 'Error viewing data!')
        return render(request, 'Guest/view_organiser.html')

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
        participant_name = request.POST.get('txtn')
        participant_email = request.POST.get('email')

        # Check if participant already exists
        participant, created = Participant.objects.get_or_create(
            email=participant_email,
            defaults={'name': participant_name}
        )

        if not created:
            # Update participant's name if they already exist
            participant.name = participant_name
            participant.save()

        request.session['participant'] = participant.id

        return redirect('view_organiser')

    return render(request, 'Guest/Event.html')

def choose_activity(request, id):
    par=request.session['participant']
    participant = get_object_or_404(Participant, id=par)
    user=User.objects.get(id=id)
    events = Event.objects.filter(is_active=True,created_by=user)
    
    if request.method == 'POST':
        for event in events:
            preference_key = f'preference_{event.id}'
            preference_value = request.POST.get(preference_key)
            # Set default value to 0 if preference_value is None
            preference_value = int(preference_value) if preference_value is not None else 0
            # Save or update the preference for each event
            ParticipantActivity.objects.update_or_create(
                    participant=participant,
                    event=event,
                    defaults={'preference': preference_value}
                )

        messages.success(request, 'You have changed your interest. Wait for results updates.')
        return redirect('home')

    # Retrieve existing preferences
    preferences = ParticipantActivity.objects.filter(participant=participant).values('event_id', 'preference')
    preferences_dict = {pref['event_id']: pref['preference'] for pref in preferences}

    return render(request, 'Guest/activity.html', {'data': events, 'participant': participant, 'preferences': preferences_dict})
