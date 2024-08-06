from AllocationAdmin.models import Event, ParticipantActivity, Participant
from django.contrib import messages
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
# from gurobipy import Model, GRB, quicksum
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, value
import ast


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


# def solve_activity_assignment(n, a, min_bounds, max_bounds, preferences):
#     # Define the model
#     model = Model("ActivityAssignment")
    
#     # Decision variables
#     x = model.addVars(n, a, vtype=GRB.BINARY, name="x")
#     y = model.addVars(a, vtype=GRB.BINARY, name="y")
    
#     # Normalize preferences
#     normalized_preferences = [[min(1, max(0, preferences[i][j])) for j in range(a)] for i in range(n)]
    
#     # Objective function
#     model.setObjective(quicksum(normalized_preferences[i][j] * x[i, j] for i in range(n) for j in range(a)), GRB.MAXIMIZE)
    
#     # Constraints
#     # Each participant must be assigned to at most one activity
#     for i in range(n):
#         model.addConstr(quicksum(x[i, j] for j in range(a)) <= 1)
    
#     # Ensure each activity j has the correct number of participants assigned within bounds
#     for j in range(a):
#         model.addConstr(min_bounds[j] * y[j] <= quicksum(x[i, j] for i in range(n)))
#         model.addConstr(quicksum(x[i, j] for i in range(n)) <= max_bounds[j] * y[j])
    
#     # Preference constraints: Ensure a participant is not assigned to activities with negative preferences
#     for i in range(n):
#         model.addConstr(quicksum(normalized_preferences[i][j] * x[i, j] for j in range(a)) >= 0)
    
#     # Ensure correct values for assigned activities
#     for j in range(a):
#         model.addConstr(quicksum(x[i, j] for i in range(n)) >= y[j])

#     # Optimize the model
#     model.optimize()
    
#     # Output results
#     assignments = []
#     for i in range(n):
#         for j in range(a):
#             if x[i, j].x > 0.5:  # Because variables are binary, this checks if they are 1
#                 assignments.append((i+1, j+1))
    
#     assigned_activities = [j+1 for j in range(a) if y[j].x > 0.5]
    
#     return assignments, assigned_activities

# @login_required
# def allocate_participants(request, ):

#         # Get the current user
#         user = request.user
        
#         # Filter events created by the logged-in user
#         events = Event.objects.filter(created_by=user, is_active=True)
        
#         # Filter participants who have shown interest in these events
#         participants = Participant.objects.filter(assigned_to__in=events).distinct()
        
#         n = participants.count()
#         a = events.count()
        
#         # Get min and max bounds for activities from POST data
#         min_bounds = [int(request.POST.get(f'min_bound_{j}')) for j in range(a)]
#         max_bounds = [int(request.POST.get(f'max_bound_{j}')) for j in range(a)]
        
#         # Retrieve preferences from ParticipantActivity model
#         preferences = []
#         for participant in participants:
#             participant_activity = ParticipantActivity.objects.get(participant=participant, event__in=events)
#             prefs = [int(x) for x in participant_activity.preferences.strip('[]').split(',')]
#             preferences.append(prefs)
        
#         # Solve the allocation problem
#         assignments, assigned_activities = solve_activity_assignment(n, a, min_bounds, max_bounds, preferences)

#         # Clear existing assignments for participants
#         Participant.objects.filter(id__in=[assignment[0] for assignment in assignments]).update(assigned_to=None)
        
#         # Save the results in the database
#         for participant_id, activity_id in assignments:
#             participant = Participant.objects.get(id=participant_id)
#             activity = Event.objects.get(id=activity_id)
            
#             # Update the participant's assigned event
#             participant.assigned_to = activity
#             participant.save()
            
#             # Update or create the ParticipantActivity record
#             ParticipantActivity.objects.update_or_create(
#                 participant=participant,
#                 event=activity,
#                 defaults={'preferences': str(preferences[participant_id-1])}
#             )
        
#         messages.success(request, "Allocation completed successfully!")
#         return redirect('list_participants', id=id)
    



def solve_activity_assignment(n, a, min_bounds, max_bounds, preferences):
    model = LpProblem("ActivityAssignment", LpMaximize)

    x = LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(a)), cat=LpBinary)
    y = LpVariable.dicts("y", (j for j in range(a)), cat=LpBinary)

    # Objective function
    model += lpSum(preferences[i][j] * x[i, j] for i in range(n) for j in range(a))

    # Each participant must be assigned to at most one activity
    for i in range(n):
        model += lpSum(x[i, j] for j in range(a)) <= 1

    # Ensure each activity j has the correct number of participants assigned within bounds
    for j in range(a):
        model += min_bounds[j] * y[j] <= lpSum(x[i, j] for i in range(n))
        model += lpSum(x[i, j] for i in range(n)) <= max_bounds[j] * y[j]

    # Ensure correct values for assigned activities
    for j in range(a):
        model += lpSum(x[i, j] for i in range(n)) >= y[j]

    model.solve()

    assignments = [(i + 1, j + 1) for i in range(n) for j in range(a) if value(x[i, j]) > 0.5]
    assigned_activities = [j + 1 for j in range(a) if value(y[j]) > 0.5]

    return assignments, assigned_activities


def allocate_participants(request):
    user = request.user
    events = Event.objects.filter(created_by=user, is_active=True)
    participants = Participant.objects.filter(is_active=True)
    
    n = participants.count()
    a = events.count()
    
    min_bounds = [event.min_participants for event in events]
    max_bounds = [event.max_participants for event in events]
    
    preferences = []
    for participant in participants:
        prefs = []
        for event in events:
            try:
                pa = ParticipantActivity.objects.get(participant=participant, event=event)
                # Convert the preferences string to a list of integers
                pref_list = ast.literal_eval(pa.preferences)
                prefs.extend(pref_list)  # Use extend to add the entire list
            except ParticipantActivity.DoesNotExist:
                prefs.extend([0] * a)  # Assuming neutral preference for missing entries
        preferences.append(prefs)
    
    assignments, assigned_activities = solve_activity_assignment(n, a, min_bounds, max_bounds, preferences)
    
    for participant_id, event_id in assignments:
        participant = participants[participant_id - 1]  # Adjust index for zero-based indexing
        event = events[event_id - 1]
        participant.assigned_to = event
        participant.save()
    
    messages.success(request, "Allocation completed successfully!")
    return redirect('view_allocation')



def view_allocation(request):
    participants = Participant.objects.all()
    return render(request, 'Organizer/allocation.html', {'participants': participants})

def solve_activity_assignment_pulp(n, a, min_bounds, max_bounds, Preferences):
    # Create the LP model
    model = LpProblem("ActivityAssignment", LpMaximize)

    # Decision variables
    x = LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(a)), cat=LpBinary)
    y = LpVariable.dicts("y", (j for j in range(a)), cat=LpBinary)

    # Normalize preferences: only consider non-negative preferences
    normalized_preferences = [[max(0, min(1, Preferences[i][j])) for j in range(a)] for i in range(n)]

    # Objective function: maximize total preference score
    model += lpSum(normalized_preferences[i][j] * x[i, j] for i in range(n) for j in range(a))

    # Constraints

    # Each participant can be assigned to at most one activity
    for i in range(n):
        model += lpSum(x[i, j] for j in range(a)) <= 1, f"Participant_{i}_Assignment"

    # Ensure each activity j has the correct number of participants assigned within bounds
    for j in range(a):
        model += min_bounds[j] * y[j] <= lpSum(x[i, j] for i in range(n)), f"Min_Participants_Activity_{j}"
        model += lpSum(x[i, j] for i in range(n)) <= max_bounds[j] * y[j], f"Max_Participants_Activity_{j}"

    # Ensure correct values for assigned activities
    for j in range(a):
        model += lpSum(x[i, j] for i in range(n)) >= y[j], f"Activity_{j}_Activation"

    # Do not assign participants to activities with negative preferences
    for i in range(n):
        for j in range(a):
            if Preferences[i][j] < 0:
                model += x[i, j] == 0, f"Negative_Preference_{i}_{j}"

    # Solve the model
    model.solve()

    # Output results
    assignments = []
    for i in range(n):
        for j in range(a):
            if x[i, j].varValue > 0.5:  # Because variables are binary, this checks if they are 1
                assignments.append((i+1, j+1))

    assigned_activities = [j+1 for j in range(a) if y[j].varValue > 0.5]

    return assignments, assigned_activities

def allocate_participants_new(request):
    user = request.user
    events = Event.objects.filter(created_by=user, is_active=True)
    participants = Participant.objects.filter(is_active=True)
    
    n = participants.count()
    a = events.count()
    
    min_bounds = [event.min_participants for event in events]
    max_bounds = [event.max_participants for event in events]
    
    preferences = []
    for participant in participants:
        prefs = []
        for event in events:
            try:
                pa = ParticipantActivity.objects.get(participant=participant, event=event)
                # Convert the preferences string to a list of integers
                pref_list = ast.literal_eval(pa.preferences)
                prefs.extend(pref_list)  # Use extend to add the entire list
            except ParticipantActivity.DoesNotExist:
                prefs.extend([0] * a)  # Assuming neutral preference for missing entries
        preferences.append(prefs)
    
    # Choose the algorithm based on a request parameter
    
    assignments, assigned_activities = solve_activity_assignment_pulp(n, a, min_bounds, max_bounds, preferences)
   
    
    for participant_id, event_id in assignments:
        participant = participants[participant_id - 1]  # Adjust index for zero-based indexing
        event = events[event_id - 1]
        participant.assigned_to_new = event
        participant.save()
    
    messages.success(request, "Allocation completed successfully!")
    return redirect('view_allocation_new')

def view_allocation_new(request):
    participants = Participant.objects.all()
    return render(request, 'Organizer/new_allocation.html', {'participants': participants})