from django.shortcuts import render,redirect
from AllocationAdmin.models import Event, ParticipantActivity
from django.contrib import messages

# Create your views here.
def index(request):
    data=Event.objects.filter(created_by=request.user,is_active=True).order_by('-created_on')
    return render(request, 'Organizer/Home.html',{'data':data})

def events(request):
    try:
        if request.method == 'POST':
        # Extract data from POST request
            event_code = request.POST.get('txtcode')
            name = request.POST.get('name')
            min_participants = int(request.POST.get('min'))
            max_participants = int(request.POST.get('max'))
            remarks = request.POST.get('remarks')

            # Validate the data if needed

            # Create and save Event instance
            event = Event(
                code=event_code,
                name=name,
                min_participants=min_participants,
                max_participants=max_participants,
                description=remarks,
                created_by=request.user,
            )
            event.save()
            return redirect('index')
        else:
            return render(request, 'Organizer/Event.html')
    except Exception as e:
        print(e)
        messages.error(request, 'The Event Code is same please try with different code.')
        return render(request, 'Organizer/Event.html')
    
def event_details(request, id):
    event = Event.objects.get(id=id)
    return render(request, 'Organizer/eventview.html', {'data': event})

def event_edit(request, id):
    event = Event.objects.get(id=id)
    if request.method == 'POST':
        # Extract data from POST request
        event_code = request.POST.get('txtcode')
        name = request.POST.get('name')
        min_participants = int(request.POST.get('min'))
        max_participants = int(request.POST.get('max'))
        remarks = request.POST.get('remarks')

        # Validate the data if needed

        # Update Event instance
        event.code = event_code
        event.name = name
        event.min_participants = min_participants
        event.max_participants = max_participants
        event.description = remarks
        event.save()
        return redirect('index')
    else:
        return render(request, 'Organizer/Event.html', {'data': event})
    

def event_delete(request, id):
    event = Event.objects.get(id=id)
    event.is_active = False
    event.save()
    return redirect('index')

def event_activate(request, id):
    event = Event.objects.get(id=id)
    event.is_active = True
    event.save()
    return redirect('index')

def list_participants(request,id):
    event = Event.objects.get(id=id)
    particpantObj=ParticipantActivity.objects.filter(activity=event)
    return render(request, 'Organizer/Status.html', {'data': particpantObj})

  
