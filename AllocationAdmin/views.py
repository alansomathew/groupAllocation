from django.shortcuts import render, redirect
from AllocationAdmin.models import Event, ParticipantActivity, Participant
from django.contrib import messages
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from gurobipy import Model, GRB, quicksum


# Create your views here.
def index(request):
    data = Event.objects.filter(created_by=request.user, is_active=True).order_by(
        "-created_on"
    )
    return render(request, "Organizer/Home.html", {"data": data})


def events(request):
    try:
        if request.method == "POST":
            # Extract data from POST request
            event_code = request.POST.get("txtcode")
            name = request.POST.get("name")
            min_participants = int(request.POST.get("min"))
            max_participants = int(request.POST.get("max"))
            remarks = request.POST.get("remarks")

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
            return redirect("index")
        else:
            return render(request, "Organizer/Event.html")
    except Exception as e:
        print(e)
        messages.error(
            request, "The Event Code is same please try with different code."
        )
        return render(request, "Organizer/Event.html")


def event_details(request, id):
    event = Event.objects.get(id=id)
    return render(request, "Organizer/eventview.html", {"data": event})


def event_edit(request, id):
    event = Event.objects.get(id=id)
    if request.method == "POST":
        # Extract data from POST request
        event_code = request.POST.get("txtcode")
        name = request.POST.get("name")
        min_participants = int(request.POST.get("min"))
        max_participants = int(request.POST.get("max"))
        remarks = request.POST.get("remarks")

        # Validate the data if needed

        # Update Event instance
        event.code = event_code
        event.name = name
        event.min_participants = min_participants
        event.max_participants = max_participants
        event.description = remarks
        event.save()
        return redirect("index")
    else:
        return render(request, "Organizer/Event.html", {"data": event})


def event_delete(request, id):
    event = Event.objects.get(id=id)
    event.is_active = False
    event.save()
    return redirect("index")


def event_activate(request, id):
    event = Event.objects.get(id=id)
    event.is_active = True
    event.save()
    return redirect("index")


def list_participants(request, id):
    event = Event.objects.get(id=id)
    participant_activities = ParticipantActivity.objects.filter(event=event)
    return render(request, "Organizer/Status.html", {"event": event, "participant_activities": participant_activities})


def solve_activity_assignment(n, a, min_bounds, max_bounds, preferences):
    # Define the model
    model = Model("ActivityAssignment")
    
    # Decision variables
    x = model.addVars(n, a, vtype=GRB.BINARY, name="x")
    y = model.addVars(a, vtype=GRB.BINARY, name="y")
    
    # Normalize preferences
    normalized_preferences = [[min(1, max(0, preferences[i][j])) for j in range(a)] for i in range(n)]
    
    # Objective function
    model.setObjective(quicksum(normalized_preferences[i][j] * x[i, j] for i in range(n) for j in range(a)), GRB.MAXIMIZE)
    
    # Constraints
    # Each participant must be assigned to at most one activity
    for i in range(n):
        model.addConstr(quicksum(x[i, j] for j in range(a)) <= 1)
    
    # Ensure each activity j has the correct number of participants assigned within bounds
    for j in range(a):
        model.addConstr(min_bounds[j] * y[j] <= quicksum(x[i, j] for i in range(n)))
        model.addConstr(quicksum(x[i, j] for i in range(n)) <= max_bounds[j] * y[j])
    
    # Preference constraints: Ensure a participant is not assigned to activities with negative preferences
    for i in range(n):
        model.addConstr(quicksum(normalized_preferences[i][j] * x[i, j] for j in range(a)) >= 0)
    
    # Ensure correct values for assigned activities
    for j in range(a):
        model.addConstr(quicksum(x[i, j] for i in range(n)) >= y[j])

    # Optimize the model
    model.optimize()
    
    # Output results
    assignments = []
    for i in range(n):
        for j in range(a):
            if x[i, j].x > 0.5:  # Because variables are binary, this checks if they are 1
                assignments.append((i+1, j+1))
    
    assigned_activities = [j+1 for j in range(a) if y[j].x > 0.5]
    
    return assignments, assigned_activities

@login_required
def allocate_participants(request, ):

        # Get the current user
        user = request.user
        
        # Filter events created by the logged-in user
        events = Event.objects.filter(created_by=user, is_active=True)
        
        # Filter participants who have shown interest in these events
        participants = Participant.objects.filter(assigned_to__in=events).distinct()
        
        n = participants.count()
        a = events.count()
        
        # Get min and max bounds for activities from POST data
        min_bounds = [int(request.POST.get(f'min_bound_{j}')) for j in range(a)]
        max_bounds = [int(request.POST.get(f'max_bound_{j}')) for j in range(a)]
        
        # Retrieve preferences from ParticipantActivity model
        preferences = []
        for participant in participants:
            participant_activity = ParticipantActivity.objects.get(participant=participant, event__in=events)
            prefs = [int(x) for x in participant_activity.preferences.strip('[]').split(',')]
            preferences.append(prefs)
        
        # Solve the allocation problem
        assignments, assigned_activities = solve_activity_assignment(n, a, min_bounds, max_bounds, preferences)

        # Clear existing assignments for participants
        Participant.objects.filter(id__in=[assignment[0] for assignment in assignments]).update(assigned_to=None)
        
        # Save the results in the database
        for participant_id, activity_id in assignments:
            participant = Participant.objects.get(id=participant_id)
            activity = Event.objects.get(id=activity_id)
            
            # Update the participant's assigned event
            participant.assigned_to = activity
            participant.save()
            
            # Update or create the ParticipantActivity record
            ParticipantActivity.objects.update_or_create(
                participant=participant,
                event=activity,
                defaults={'preferences': str(preferences[participant_id-1])}
            )
        
        messages.success(request, "Allocation completed successfully!")
        return redirect('list_participants', id=id)
    
   


def view_allocation(request, id):
    # Get the event for which the allocation is to be displayed
    event = Event.objects.get(id=id)
    
    # Fetch all participants and their assigned activities
    participant_activities = ParticipantActivity.objects.filter(event=event)
    
    # Create a dictionary to hold participant details and their preferences
    participant_data = []
    for participant_activity in participant_activities:
        participant = participant_activity.participant
        participant_data.append({
            'name': participant.name,
            'preferences': participant_activity.preferences,
            'assigned_to': participant_activity.event.name
        })
    
    return render(request, 'Organizer/view_allocation.html', {
        'event': event,
        'participant_data': participant_data
    })


