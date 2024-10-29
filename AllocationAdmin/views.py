from AllocationAdmin.models import Event, ParticipantActivity, Participant
from itertools import product
from django.contrib import messages
from django.shortcuts import render, redirect, get_object_or_404
from django.db import transaction
from django.contrib.auth.decorators import login_required
# from gurobipy import Model, GRB, quicksum
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, LpBinary
import pulp
import ast


# Create your views here.
def index(request):
    data = Event.objects.filter(created_by=request.user,).order_by(
        "-created_on"
    )
    user = request.user
    # Assuming `is_updated` is a field in your custom user model
    is_updated = request.user.is_updated
    # Assuming `is_updated_new` is a field in your custom user model
    is_updated_new = request.user.is_updated_new
    # Assuming `is_updated_max` is a field in your custom user model
    is_updated_max = request.user.is_updated_max

    context = {
        "data": data,
        "is_updated": is_updated,
        "is_updated_new": is_updated_new,
        # Assuming `is_updated_max` is a field in your custom user model
        "is_updated_max": is_updated_max,
    }

    return render(request, "Organizer/Home.html", context)


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

# start event and stop event


def start_event(request, event_id):
    event = Event.objects.get(id=event_id)
    event.is_active = True
    event.save()
    messages.success(request, 'Event started successfully.')
    return redirect('event_details', id=event_id)


def stop_event(request, event_id):
    event = Event.objects.get(id=event_id)
    event.is_active = False
    event.save()
    messages.success(request, 'Event stopped successfully.')
    return redirect('event_details', id=event_id)


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


# Function to solve activity assignment using linear programming (as given in your original code)
def solve_activity_assignment(n, a, min_bounds, max_bounds, Preferences, participant_names, event_names):
    # Define the model
    model = LpProblem("ActivityAssignment", LpMaximize)

    # Decision variables
    x = LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(a)), cat=LpBinary)
    y = LpVariable.dicts("y", (j for j in range(a)), cat=LpBinary)

    # Normalize preferences
    normalized_preferences = [
        [min(1, max(0, Preferences[i][j])) for j in range(a)] for i in range(n)
    ]

    # Objective function
    model += lpSum(normalized_preferences[i][j] * x[i, j] for i in range(n) for j in range(a))

    # Constraints
    # Each participant must be assigned to at most one activity
    for i in range(n):
        model += lpSum(x[i, j] for j in range(a)) <= 1

    # Ensure each activity j has the correct number of participants assigned within bounds
    for j in range(a):
        model += min_bounds[j] * y[j] <= lpSum(x[i, j] for i in range(n))
        model += lpSum(x[i, j] for i in range(n)) <= max_bounds[j] * y[j]

    # Preference constraints: Ensure a participant is not assigned to activities with negative preferences
    for i in range(n):
        model += lpSum(normalized_preferences[i][j] * x[i, j] for j in range(a)) >= 0

    # Ensure correct values for assigned activities
    for j in range(a):
        model += lpSum(x[i, j] for i in range(n)) >= y[j]

    # Optimize the model
    model.solve()

    # Output results
    assignments = []
    for i in range(n):
        for j in range(a):
            if value(x[i, j]) > 0.5:  # Because variables are binary, this checks if they are 1
                assignments.append((i, j))

    assigned_activities = [j for j in range(a) if value(y[j]) > 0.5]

    return assignments, assigned_activities


# Function to compute feasible R values for each activity
def compute_R_values(assignments, min_bounds, max_bounds):
    """
    Compute the R values for each activity.
    R represents the range of feasible numbers of participants that can be moved from each activity,
    considering both minimum and maximum bounds.
    """
    R = {}
    for activity in set(assignments.values()):
        current_count = len([k for k in assignments.values() if k == activity])
        min_count = min_bounds[activity]
        max_count = max_bounds[activity]

        # Generate R values considering both lower and upper bounds
        R[activity] = [0]
        for h in range(1, current_count + 1):
            if (current_count - h) >= min_count and (current_count - h) <= max_count:
                R[activity].append(h)
    
    return R

def find_b_Wbar_r(R, c):
    """
    Calculate breakpoint 'b', total weight 'Wbar', and maximum weight 'r' using R values.
    Parameters:
    R: Dictionary where keys are activity IDs and values are lists representing feasible participant moves.
    c: Capacity for the target activity.
    """
    activities = list(R.keys())  # Get the list of activity IDs
    k = len(activities)
    b = 1
    Wbar = 0

    while b <= k:
        # Get activities up to the b-th activity
        beta_activities = activities[:b]
        alpha_activities = activities[b:]

        # Calculate sums for beta and alpha
        beta_sum = sum(max(R[act]) for act in beta_activities)  # Sum of maximum values for the first 'b' activities
        alpha_sum = sum(min(R[act]) for act in alpha_activities)  # Sum of minimum values for remaining activities
        total_sum = beta_sum + alpha_sum

        # Check if total sum exceeds the capacity c
        if total_sum > c:
            break
        Wbar = beta_sum  # Update Wbar if the sum doesn't exceed capacity
        b += 1

    r = max(max(R[act]) for act in activities)  # Calculate the maximum value across all R values
    return b, Wbar, r


def algorithm_mcssp(R, c):
    """Implement MCSSP using dynamic programming to determine feasible reassignments using R values."""
    k = len(R)
    b, Wbar, r = find_b_Wbar_r(R, c)

    S = [[0] * (c + r + 1) for _ in range(k + 1)]
    for participant_count in range(c - r + 1, c):
        S[b - 1][participant_count] = 0
    for participant_count in range(c + 1, c + r + 1):
        S[b - 1][participant_count] = 1
    S[b - 1][Wbar] = b

    for t in range(b, k + 1):
        for participant_count in range(c - r + 1, c + r + 1):
            S[t][participant_count] = S[t - 1][participant_count]

        max_weight = max(R[t - 1])
        for participant_count in range(c - r + 1, c + 1):
            for i in R[t - 1]:
                new_count = participant_count + i - min(R[t - 1])
                if 0 <= new_count < len(S[t]):
                    S[t][new_count] = max(S[t][new_count], S[t - 1][participant_count])

        for participant_count in range(c + max_weight, c, -1):
            for j in range(S[t - 1][participant_count], S[t][participant_count]):
                for i in R[j - 1]:
                    new_count = participant_count + i - max(R[j - 1])
                    if 0 <= new_count < len(S[t]):
                        S[t][new_count] = max(S[t][new_count], j)

    column_C_values = [S[row][c] for row in range(b - 1, k + 1)]
    optimal_solution = max(column_C_values)
    return optimal_solution


@login_required
def allocate_participants_to_activities(request):
    # Get the active events created by the current user (organizer)
    events = Event.objects.filter(is_active=True, created_by=request.user)
    participants = Participant.objects.filter(participantactivity__event__in=events).distinct()

    n = participants.count()
    a = events.count()

    if n == 0 or a == 0:
        messages.warning(request, "No participants or events available for allocation.")
        return redirect('view_allocation')

    min_bounds = list(events.values_list('min_participants', flat=True))
    max_bounds = list(events.values_list('max_participants', flat=True))

    Preferences = []
    participant_names = []
    event_names = list(events.values_list('name', flat=True))

    for participant in participants:
        preferences = []
        participant_names.append(participant.name)
        for event in events:
            activity_preference = ParticipantActivity.objects.filter(participant=participant, event=event).first()
            preferences.append(activity_preference.preference if activity_preference else 0)
        Preferences.append(preferences)

    # Solve the assignment problem
    assignments, assigned_activities = solve_activity_assignment(
        n, a, min_bounds, max_bounds, Preferences, participant_names, event_names
    )

    # Update participant assignments in the database
    with transaction.atomic():
        for participant_idx, event_idx in assignments:
            participant = participants[participant_idx]
            event = events[event_idx]
            participant.assigned_to = event
            participant.save()

    # Display messages regarding allocation results
    if len(assignments) == n:
        messages.success(request, "All participants are involved in the allocation.")
    else:
        messages.warning(request, "Not all participants are involved in the allocation.")

    if len(assigned_activities) == a:
        messages.success(request, "All events have at least one participant.")
    else:
        messages.warning(request, "Not all events have participants.")

    # Calculate total preferences for each event
    total_preferences = [0] * a
    for i in range(n):
        for j in range(a):
            total_preferences[j] += Preferences[i][j]

    max_pref_value = max(total_preferences)
    min_pref_value = min(total_preferences)

    max_pref_events = [event_names[j] for j in range(a) if total_preferences[j] == max_pref_value]
    min_pref_events = [event_names[j] for j in range(a) if total_preferences[j] == min_pref_value]

    # Display Most and Least Interested Activities
    messages.info(request, f"Most Interested Activity: {', '.join(max_pref_events)}")
    messages.info(request, f"Least Interested Activity: {', '.join(min_pref_events)}")

    return redirect('view_allocation')


@login_required
def view_allocation(request):
    try:
        # Get the events created by the current user (organizer)
        events = Event.objects.filter(created_by=request.user)
        participants = Participant.objects.filter(participantactivity__event__in=events).distinct()

        # Prepare the list of events for indexing
        events_list = list(events)
        a = len(events_list)
        n = len(participants)

        if n == 0 or a == 0:
            messages.warning(request, "No participants or events available for viewing.")
            return redirect('home')
        
        min_bounds = list(events.values_list('min_participants', flat=True))
        max_bounds = list(events.values_list('max_participants', flat=True))

        # Prepare the Preferences matrix and assignment dictionary
        Preferences = []
        assignment_dict = {}
        assignments = []  # Store current assignments

        for idx, participant in enumerate(participants):
            preferences = []
            assigned_event = participant.assigned_to
            assigned_event_idx = events_list.index(assigned_event) if assigned_event in events_list else None
            if assigned_event_idx is not None:
                assignment_dict[idx] = assigned_event_idx
                assignments.append((idx, assigned_event_idx))  # Add current assignments

            for event in events_list:
                activity_preference = ParticipantActivity.objects.filter(participant=participant, event=event).first()
                preferences.append(activity_preference.preference if activity_preference else 0)
            Preferences.append(preferences)

        participant_names = [participant.name for participant in participants]
        event_names = [event.name for event in events]

        # Core Stability, Individual Stability, and Individual Rationality Checks
        individual_stability_violations = []
        individual_rationality_violations = []
        core_stability_violations = []

        # **Individual Stability Check**
        for i in range(n):
            assigned_event_idx = assignment_dict.get(i, None)
            if assigned_event_idx is None:
                continue

            # Check Individual Stability
            for j in range(a):
                if Preferences[i][j] > Preferences[i][assigned_event_idx] and j != assigned_event_idx:
                    individual_stability_violations.append(
                        f"{participant_names[i]} can improve by switching from {event_names[assigned_event_idx]} to {event_names[j]}."
                    )

            # **Individual Rationality Check**
            if Preferences[i][assigned_event_idx] <= 0:
                individual_rationality_violations.append(
                    f"{participant_names[i]} is not individually rational in their assigned {event_names[assigned_event_idx]}."
                )

        # **Core Stability Check**
        R_values = compute_R_values(assignment_dict, min_bounds, max_bounds)
        for i in range(n):
            assigned_event_idx = assignment_dict.get(i, None)
            for j in range(a):
                if Preferences[i][j] > Preferences[i][assigned_event_idx] and j != assigned_event_idx:
                    # Determine if the coalition can move to activity j
                    if algorithm_mcssp(R_values, max_bounds[j]) > 0:
                        core_stability_violations.append(
                            f"{participant_names[i]} and others can jointly benefit by switching to {event_names[j]}."
                        )

        # **Messages for Stability Violations**
        if individual_stability_violations:
            for violation in individual_stability_violations:
                messages.warning(request, violation)
            messages.error(request, "The assignment is not individually stable.")
        else:
            messages.success(request, "The assignment is individually stable.")

        if core_stability_violations:
            for violation in core_stability_violations:
                messages.warning(request, violation)
            messages.error(request, "The assignment is not core stable.")
        else:
            messages.success(request, "The assignment is core stable.")

        if individual_rationality_violations:
            for violation in individual_rationality_violations:
                messages.warning(request, violation)
            messages.error(request, "The assignment is not individually rational.")
        else:
            messages.success(request, "The assignment is individually rational.")

        return render(request, 'Organizer/allocation.html', {
            'participants': participants,
            'individual_stability_violations': individual_stability_violations,
            'core_stability_violations': core_stability_violations,
            'individual_rationality_violations': individual_rationality_violations,
        })

    except Exception as e:
        print(e)
        messages.error(request, 'Error viewing allocations!')
        return render(request, 'Organizer/allocation.html')


def compute_R_values_new(assignments, min_bounds, max_bounds):
    """
    Compute the R values for each activity.
    R represents the range of feasible numbers of participants that can be moved from each activity,
    considering both minimum and maximum bounds.
    """
    R = {}
    for activity in set(assignments.values()):
        current_count = len([k for k in assignments.values() if k == activity])
        min_count = min_bounds[activity]
        max_count = max_bounds[activity]

        # Generate R values considering both lower and upper bounds
        R[activity] = [0]
        for h in range(1, current_count + 1):
            if (current_count - h) >= min_count and (current_count - h) <= max_count:
                R[activity].append(h)

    return R

def find_b_Wbar_r_new(R, c):
    """Calculate breakpoint 'b', total weight 'Wbar', and maximum weight 'r' using R values."""
    activities = list(R.keys())  # Get the list of activities (keys of R)
    k = len(activities)
    b = 1
    Wbar = 0

    while b <= k:
        # Calculate sums for beta and alpha using activity keys
        beta_activities = activities[:b]  # Activities up to b (exclusive)
        alpha_activities = activities[b:]  # Activities from b onward

        beta_sum = sum(max(R[activity]) for activity in beta_activities)  # Sum of maximum values for first b activities
        alpha_sum = sum(min(R[activity]) for activity in alpha_activities)  # Sum of minimum values for remaining activities
        total_sum = beta_sum + alpha_sum

        # Check if total sum exceeds the capacity c
        if total_sum > c:
            break

        Wbar = beta_sum  # Update Wbar if the sum doesn't exceed capacity
        b += 1

    # Calculate r as the maximum value across all R values
    r = max(max(R[activity]) for activity in activities)
    return b, Wbar, r


def algorithm_mcssp_new(R, c):
    """Implement MCSSP using dynamic programming to determine feasible reassignments using R values."""
    k = len(R)
    b, Wbar, r = find_b_Wbar_r_new(R, c)

    S = [[0] * (c + r + 1) for _ in range(k + 1)]
    for participant_count in range(c - r + 1, c):
        S[b - 1][participant_count] = 0
    for participant_count in range(c + 1, c + r + 1):
        S[b - 1][participant_count] = 1
    S[b - 1][Wbar] = b

    for t in range(b, k + 1):
        for participant_count in range(c - r + 1, c + r + 1):
            S[t][participant_count] = S[t - 1][participant_count]

        max_weight = max(R[t - 1])
        for participant_count in range(c - r + 1, c + 1):
            for i in R[t - 1]:
                new_count = participant_count + i - min(R[t - 1])
                if 0 <= new_count < len(S[t]):
                    S[t][new_count] = max(S[t][new_count], S[t - 1][participant_count])

        for participant_count in range(c + max_weight, c, -1):
            for j in range(S[t - 1][participant_count], S[t][participant_count]):
                for i in R[j - 1]:
                    new_count = participant_count + i - max(R[j - 1])
                    if 0 <= new_count < len(S[t]):
                        S[t][new_count] = max(S[t][new_count], j)

    column_C_values = [S[row][c] for row in range(b - 1, k + 1)]
    optimal_solution = max(column_C_values)
    return optimal_solution

def solve_activity_assignment_pulp(n, a, min_bounds, max_bounds, Preferences, participants, events):
    model = LpProblem("ActivityAssignment", LpMaximize)

    # Decision Variables
    x = LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(a)), cat=LpBinary)
    y = LpVariable.dicts("y", (j for j in range(a)), cat=LpBinary)

    # Normalize Preferences
    normalized_preferences = [
        [max(0, min(1, Preferences[i][j])) for j in range(a)] for i in range(n)
    ]
    model += lpSum(normalized_preferences[i][j] * x[i, j] for i in range(n) for j in range(a))

    # Constraints
    for i in range(n):
        model += lpSum(x[i, j] for j in range(a)) <= 1, f"Participant_{i}_Assignment"
    
    for j in range(a):
        model += min_bounds[j] * y[j] <= lpSum(x[i, j] for i in range(n)), f"Min_Participants_Activity_{j}"
        model += lpSum(x[i, j] for i in range(n)) <= max_bounds[j] * y[j], f"Max_Participants_Activity_{j}"
    
    for j in range(a):
        model += lpSum(x[i, j] for i in range(n)) >= y[j], f"Activity_{j}_Activation"
    
    for i in range(n):
        for j in range(a):
            if Preferences[i][j] < 0:
                model += x[i, j] == 0, f"Negative_Preference_{i}_{j}"

    # Solve the model
    model.solve()

    # Collect assignments and assigned activities
    assignments = [(i, j) for i in range(n) for j in range(a) if x[i, j].varValue > 0.5]
    assigned_activities = [j for j in range(a) if y[j].varValue > 0.5]

    # Mapping indices to names
    participant_names = [p.name for p in participants]
    event_names = [e.name for e in events]

    # Compute R values
    assignment_dict = {i: j for i, j in assignments}
    R_values = compute_R_values_new(assignment_dict, min_bounds, max_bounds)

    # Calculate Wbar and r using find_b_Wbar_r_new function
    for target_activity in range(a):
        b, Wbar, r = find_b_Wbar_r_new(R_values, max_bounds[target_activity])
        print(f"For target activity {event_names[target_activity]}: b = {b}, Wbar = {Wbar}, r = {r}")

    return assignments, assigned_activities


# Updated View Allocation
@login_required
def allocate_participants_new(request):
    events = Event.objects.filter(is_active=True, created_by=request.user)
    participants = Participant.objects.filter(participantactivity__event__in=events).distinct()

    n = participants.count()
    a = events.count()

    if n == 0 or a == 0:
        messages.warning(request, "No participants or events available for allocation.")
        return redirect('view_allocation_new')

    min_bounds = list(events.values_list('min_participants', flat=True))
    max_bounds = list(events.values_list('max_participants', flat=True))

    Preferences = []
    participant_names = []
    event_names = list(events.values_list('name', flat=True))

    for participant in participants:
        preferences = []
        participant_names.append(participant.name)
        for event in events:
            activity_preference = ParticipantActivity.objects.filter(participant=participant, event=event).first()
            preferences.append(activity_preference.preference if activity_preference else 0)
        Preferences.append(preferences)

    # Solve the assignment problem
    assignments, assigned_activities = solve_activity_assignment_pulp(
        n, a, min_bounds, max_bounds, Preferences, participants, events
    )

    # Update participant assignments in the database
    with transaction.atomic():
        for participant_idx, event_idx in assignments:
            participant = participants[participant_idx]
            event = events[event_idx]
            participant.assigned_to_new = event
            participant.save()

    # Display Most and Least Interested Activities
    messages.success(request, "Assignment is successfully calculated.")
    
    return redirect('view_allocation_new')


@login_required
def view_allocation_new(request):
    try:
        # Get the events created by the current user (organizer)
        events = Event.objects.filter(created_by=request.user)

        # Filter participants who have given preferences to any of these events
        participants = Participant.objects.filter(
            participantactivity__event__in=events
        ).distinct()

        # Get the number of participants and events
        n = participants.count()
        a = events.count()

        if n == 0 or a == 0:
            messages.warning(request, "No participants or events available for viewing.")
            return redirect('home')

        # Get the min and max bounds for events
        min_bounds = list(events.values_list('min_participants', flat=True))
        max_bounds = list(events.values_list('max_participants', flat=True))

        # Prepare the Preferences matrix and assignment dictionary
        Preferences = []
        assignment_dict = {}
        assignments = []

        for idx, participant in enumerate(participants):
            preferences = []
            assigned_event = participant.assigned_to_new
            assigned_event_idx = list(events).index(assigned_event) if assigned_event in events else None
            if assigned_event_idx is not None:
                assignment_dict[idx] = assigned_event_idx  # Use a scalar value here
                assignments.append((idx, assigned_event_idx))

            for event in events:
                activity_preference = ParticipantActivity.objects.filter(participant=participant, event=event).first()
                preferences.append(activity_preference.preference if activity_preference else 0)
            Preferences.append(preferences)

        # Initialize lists to hold stability violations
        individual_stability_violations = []
        individual_rationality_violations = []
        core_stability_violations = []

        participant_names = [p.name for p in participants]
        event_names = [e.name for e in events]

        # **Individual Stability Check**
        for i in range(n):
            assigned_event_idx = assignment_dict.get(i)
            if assigned_event_idx is None:
                continue

            # Check Individual Stability
            for j in range(a):
                if Preferences[i][j] > Preferences[i][assigned_event_idx] and j != assigned_event_idx:
                    individual_stability_violations.append(
                        f"{participant_names[i]} can improve by switching from {event_names[assigned_event_idx]} to {event_names[j]}."
                    )

            # **Individual Rationality Check**
            if Preferences[i][assigned_event_idx] <= 0:
                individual_rationality_violations.append(
                    f"{participant_names[i]} is not individually rational in their assigned {event_names[assigned_event_idx]}."
                )

        # **Core Stability Check**
        R_values = compute_R_values_new(assignment_dict, min_bounds, max_bounds)
        for i in range(n):
            assigned_event_idx = assignment_dict.get(i)
            for j in range(a):
                if Preferences[i][j] > Preferences[i][assigned_event_idx] and j != assigned_event_idx:
                    # Determine if the coalition can move to activity j
                    if algorithm_mcssp_new(R_values, max_bounds[j]) > 0:
                        core_stability_violations.append(
                            f"{participant_names[i]} and others can jointly benefit by switching to {event_names[j]}."
                        )

        # **Messages for Stability Violations**
        if individual_stability_violations:
            for violation in individual_stability_violations:
                messages.warning(request, violation)
            messages.error(request, "The assignment is not individually stable.")
        else:
            messages.success(request, "The assignment is individually stable.")

        if core_stability_violations:
            for violation in core_stability_violations:
                messages.warning(request, violation)
            messages.error(request, "The assignment is not core stable.")
        else:
            messages.success(request, "The assignment is core stable.")

        if individual_rationality_violations:
            for violation in individual_rationality_violations:
                messages.warning(request, violation)
            messages.error(request, "The assignment is not individually rational.")
        else:
            messages.success(request, "The assignment is individually rational.")

        return render(request, 'Organizer/new_allocation.html', {
            'participants': participants,
            'individual_stability_violations': individual_stability_violations,
            'core_stability_violations': core_stability_violations,
            'individual_rationality_violations': individual_rationality_violations,
        })

    except Exception as e:
        print(e)
        messages.error(request, 'Error viewing allocations!')
        return render(request, 'Organizer/new_allocation.html')


def edit_allocation(request):
    events = Event.objects.filter(is_active=True, created_by=request.user)
    participants = Participant.objects.filter(participantactivity__event__in=events).distinct()

    new_allocations = {}  # Dictionary to store the new allocations
    activity_counts = {activity.id: 0 for activity in events}  # Track the number of participants assigned to each activity

    if request.method == 'POST':
        # Process the form submission for new allocations
        for participant in participants:
            activity_id = request.POST.get(f'activity_{participant.id}')
            if activity_id:
                activity_id = int(activity_id)
                if activity_id in new_allocations:
                    new_allocations[activity_id].append(participant)
                else:
                    new_allocations[activity_id] = [participant]
                activity_counts[activity_id] += 1

        # Check for capacity issues
        for activity_id, count in activity_counts.items():
            activity = Event.objects.get(id=activity_id)
            if count > activity.max_participants:
                messages.warning(
                    request,
                    f"Activity '{activity.name}' capacity exceeded. Maximum capacity is {activity.max_participants}. Currently allocated: {count}."
                )

        # Update the participants' activity assignments
        for participant in participants:
            activity_id = request.POST.get(f'activity_{participant.id}')
            if activity_id:
                activity = Event.objects.get(id=int(activity_id))
                participant.assigned_to = activity
            else:
                participant.assigned_to = None
            participant.save()

        # Update the `is_updated` flag for the user
        user = request.user
        user.is_updated = True
        user.save()

        messages.success(request, "Allocation updated successfully.")
        return redirect('view_allocation')

    # Check individual stability for each participant's current and new allocation
    individual_stability_violations = []
    for participant in participants:
        current_event = participant.assigned_to
        participant_preferences = ParticipantActivity.objects.filter(participant=participant)

        # Get the current preferred activity (if any)
        preferred_event = max(participant_preferences, key=lambda x: x.preference, default=None)

        # Check if switching would result in a better preference
        if preferred_event and current_event != preferred_event.event:
            if preferred_event.preference > participant_preferences.filter(event=current_event).first().preference:
                individual_stability_violations.append(
                    f"Participant {participant.name} can improve by switching from {current_event.name if current_event else 'None'} to {preferred_event.event.name}."
                )

    # print(individual_stability_violations)

    if individual_stability_violations:
        for violation in individual_stability_violations:
            print(violation)
            messages.warning(request, violation)

    context = {
        'events': events,
        'participants': participants,
    }
    return render(request, 'Organizer/modify.html', context)


def edit_allocation_new(request):
    event = Event.objects.filter(is_active=True, created_by=request.user)
    participants = Participant.objects.filter(participantactivity__event__in=event).distinct()
    
    new_allocations = {}  # Dictionary to store new allocations
    activity_counts = {activity.id: 0 for activity in event}  # Track how many participants are assigned to each event

    if request.method == 'POST':
        # Process the form submission for new allocations
        for participant in participants:
            activity_id = request.POST.get(f'activity_{participant.id}')
            if activity_id:
                activity_id = int(activity_id)
                if activity_id in new_allocations:
                    new_allocations[activity_id].append(participant)
                else:
                    new_allocations[activity_id] = [participant]
                activity_counts[activity_id] += 1

        # Check for capacity issues
        for activity_id, count in activity_counts.items():
            activity = Event.objects.get(id=activity_id)
            if count > activity.max_participants:
                messages.warning(request, f"Activity '{activity.name}' capacity exceeded. Maximum capacity is {activity.max_participants}. Currently allocated: {count}.")

        # Update the participants' activity assignments
        for participant in participants:
            activity_id = request.POST.get(f'activity_{participant.id}')
            if activity_id:
                activity = Event.objects.get(id=int(activity_id))
                participant.assigned_to_new = activity
            else:
                participant.assigned_to_new = None
            participant.save()

        user = request.user
        user.is_updated_new = True
        user.save()

        messages.success(request, "Allocation updated successfully.")
        return redirect('view_allocation_new')

    # Individual Stability Check (before saving any updates)
    individual_stability_violations = []
    for participant in participants:
        current_event = participant.assigned_to_new
        participant_preferences = ParticipantActivity.objects.filter(participant=participant)

        # Get the current preferred event (if any)
        preferred_event = max(participant_preferences, key=lambda x: x.preference, default=None)

        # Check if switching would result in a better preference
        if preferred_event and current_event != preferred_event.event:
            if preferred_event.preference > participant_preferences.filter(event=current_event).first().preference:
                individual_stability_violations.append(
                    f"Participant {participant.name} can improve by switching from {current_event.name if current_event else 'None'} to {preferred_event.event.name}."
                )

    if individual_stability_violations:
        for violation in individual_stability_violations:
            messages.warning(request, violation)

    context = {
        'event': event,
        'activities': event,
        'participants': participants,
    }
    return render(request, 'Organizer/modify_allocation.html', context)



# MCSSP Helper Functions
def compute_R_values_max(assignments, min_bounds, max_bounds):
    """
    Compute the R values for each activity.
    R represents the range of feasible numbers of participants that can be moved from each activity,
    considering both minimum and maximum bounds.
    """
    R = {}
    for activity in set(assignments.values()):
        current_count = len([k for k in assignments.values() if k == activity])
        min_count = min_bounds[activity]
        max_count = max_bounds[activity]

        # Generate R values considering both lower and upper bounds
        R[activity] = [0]
        for h in range(1, current_count + 1):
            if (current_count - h) >= min_count and (current_count - h) <= max_count:
                R[activity].append(h)
    
    return R

def find_b_Wbar_r_max(R, c):
    """Calculate breakpoint 'b', total weight 'Wbar', and maximum weight 'r' using R values."""
    k = len(R)
    b = 1
    Wbar = 0
    while b <= k:
        # Calculate sums for beta and alpha
        beta_sum = sum(max(R[cls]) for cls in range(b))  # Sum of maximum values for first b activities
        alpha_sum = sum(min(R[cls]) for cls in range(b, k))  # Sum of minimum values for remaining activities
        total_sum = beta_sum + alpha_sum

        # Check if total sum exceeds the capacity c
        if total_sum > c:
            break
        Wbar = beta_sum  # Update Wbar if the sum doesn't exceed capacity
        b += 1

    r = max(max(R[i]) for i in range(k))  # Calculate the maximum value across all R values
    return b, Wbar, r

def algorithm_mcssp_max(R, c):
    """Implement MCSSP using dynamic programming to determine feasible reassignments using R values."""
    k = len(R)
    b, Wbar, r = find_b_Wbar_r_max(R, c)

    S = [[0] * (c + r + 1) for _ in range(k + 1)]
    for participant_count in range(c - r + 1, c):
        S[b - 1][participant_count] = 0
    for participant_count in range(c + 1, c + r + 1):
        S[b - 1][participant_count] = 1
    S[b - 1][Wbar] = b

    for t in range(b, k + 1):
        for participant_count in range(c - r + 1, c + r + 1):
            S[t][participant_count] = S[t - 1][participant_count]

        max_weight = max(R[t - 1])
        for participant_count in range(c - r + 1, c + 1):
            for i in R[t - 1]:
                new_count = participant_count + i - min(R[t - 1])
                if 0 <= new_count < len(S[t]):
                    S[t][new_count] = max(S[t][new_count], S[t - 1][participant_count])

        for participant_count in range(c + max_weight, c, -1):
            for j in range(S[t - 1][participant_count], S[t][participant_count]):
                for i in R[j - 1]:
                    new_count = participant_count + i - max(R[j - 1])
                    if 0 <= new_count < len(S[t]):
                        S[t][new_count] = max(S[t][new_count], j)

    column_C_values = [S[row][c] for row in range(b - 1, k + 1)]
    optimal_solution = max(column_C_values)
    return optimal_solution

# Activity Allocation Algorithm
def solve_activity_assignment_max(n, a, min_bounds, max_bounds, Preferences, participants, events):
    # Create the LP problem
    prob = LpProblem("ActivityAssignment", LpMaximize)

    # Decision variables
    # x[i][j] = 1 if participant i is assigned to activity j
    x = LpVariable.dicts("x", (range(n), range(a)), cat='Binary')
    # y[j] = 1 if activity j is assigned
    y = LpVariable.dicts("y", range(a), cat='Binary')

    # Objective function: Maximize total preference sum
    prob += lpSum(Preferences[i][j] * x[i][j] for i in range(n) for j in range(a)), "TotalPreferenceSum"

    # Constraints
    for i in range(n):
        prob += lpSum(x[i][j] for j in range(a)) <= 1, f"Participant_{i}_Assignment"
    for j in range(a):
        prob += min_bounds[j] * y[j] <= lpSum(x[i][j] for i in range(n)), f"Min_Participants_Activity_{j}"
        prob += lpSum(x[i][j] for i in range(n)) <= max_bounds[j] * y[j], f"Max_Participants_Activity_{j}"
    for j in range(a):
        prob += lpSum(x[i][j] for i in range(n)) >= y[j], f"Activity_{j}_Activation"

    # Solve the problem
    prob.solve()

    # Output results
    assignments = [(i, j) for i in range(n) for j in range(a) if value(x[i][j]) > 0.5]
    assigned_activities = [j for j in range(a) if value(y[j]) > 0.5]

    return assignments, assigned_activities

@login_required
def allocate_activities_max(request):
    try:
        # Get events and participants
        events = Event.objects.filter(created_by=request.user)
        participants = Participant.objects.filter(participantactivity__event__in=events).distinct()

        # Prepare data for optimization
        n = participants.count()
        a = events.count()
        if n == 0 or a == 0:
            messages.warning(request, "No participants or events available for allocation.")
            return redirect('view_allocation_max')

        min_bounds = list(events.values_list('min_participants', flat=True))
        max_bounds = list(events.values_list('max_participants', flat=True))

        Preferences = []
        participant_names = []
        event_names = list(events.values_list('name', flat=True))

        for participant in participants:
            preferences = []
            participant_names.append(participant.name)
            for event in events:
                activity_preference = ParticipantActivity.objects.filter(participant=participant, event=event).first()
                preferences.append(activity_preference.preference if activity_preference else 0)
            Preferences.append(preferences)

        # Solve the assignment problem
        assignments, assigned_activities = solve_activity_assignment_max(
            n, a, min_bounds, max_bounds, Preferences, participants, events
        )

        # Save the assignment results back to the database
        with transaction.atomic():
            for participant_idx, event_idx in assignments:
                participant = participants[participant_idx]
                event = events[event_idx]
                participant.assigned_to_max = event
                participant.save()

        # Core Stability, Individual Stability, and Rationality Checks
        individual_stability_violations = []
        core_stability_violations = []
        individual_rationality_violations = []

        assignment_dict = {i: j for i, j in assignments}
        R_values = compute_R_values_max(assignment_dict, min_bounds, max_bounds)

        for i in range(n):
            assigned_event_idx = assignment_dict.get(i)
            if assigned_event_idx is None:
                continue

            # Check Individual Stability
            for j in range(a):
                if Preferences[i][j] > Preferences[i][assigned_event_idx] and j != assigned_event_idx:
                    individual_stability_violations.append(
                        f"{participant_names[i]} can improve by switching from {event_names[assigned_event_idx]} to {event_names[j]}."
                    )

            # Check Individual Rationality
            if Preferences[i][assigned_event_idx] <= 0:
                individual_rationality_violations.append(
                    f"{participant_names[i]} is not individually rational in their assigned {event_names[assigned_event_idx]}."
                )

            # Check Core Stability using MCSSP
            for j in range(a):
                if Preferences[i][j] > Preferences[i][assigned_event_idx] and j != assigned_event_idx:
                    if algorithm_mcssp_max(R_values, max_bounds[j]) > 0:
                        core_stability_violations.append(
                            f"{participant_names[i]} and others can jointly benefit by switching to {event_names[j]}."
                        )

        # Display results of the allocation and stability checks
        if individual_stability_violations:
            for violation in individual_stability_violations:
                messages.warning(request, violation)
            messages.error(request, "The assignment is not individually stable.")
        else:
            messages.success(request, "The assignment is individually stable.")

        if core_stability_violations:
            for violation in core_stability_violations:
                messages.warning(request, violation)
            messages.error(request, "The assignment is not core stable.")
        else:
            messages.success(request, "The assignment is core stable.")

        if individual_rationality_violations:
            for violation in individual_rationality_violations:
                messages.warning(request, violation)
            messages.error(request, "The assignment is not individually rational.")
        else:
            messages.success(request, "The assignment is individually rational.")

        # Redirect to view allocation page
        return redirect('view_allocation_max')

    except Exception as e:
        print(e)
        messages.error(request, 'Error during the allocation process!')
        return redirect('view_allocation_max')


@login_required
def view_allocation_max(request):
    try:
        # Get the events created by the current user (organizer)
        events = Event.objects.filter(created_by=request.user)

        # Filter participants who have given preferences to any of these events
        participants = Participant.objects.filter(
            participantactivity__event__in=events
        ).distinct()

        # Get the number of participants and events
        n = participants.count()
        a = events.count()

        if n == 0 or a == 0:
            messages.warning(request, "No participants or events available for viewing.")
            return redirect('home')

        # Get the min and max bounds for events
        min_bounds = list(events.values_list('min_participants', flat=True))
        max_bounds = list(events.values_list('max_participants', flat=True))

        # Prepare the Preferences matrix and assignment dictionary
        Preferences = []
        assignment_dict = {}
        assignments = []  # Store current assignments

        # Gather the preferences and assignments for each participant
        for idx, participant in enumerate(participants):
            preferences = []
            # Assuming "assigned_to_max" holds the new assignment
            assigned_event = participant.assigned_to_max
            assigned_event_idx = list(events).index(assigned_event) if assigned_event in events else None
            if assigned_event_idx is not None:
                assignment_dict[idx] = assigned_event_idx
                assignments.append((idx, assigned_event_idx))

            for event in events:
                activity_preference = ParticipantActivity.objects.filter(
                    participant=participant, event=event).first()
                preferences.append(activity_preference.preference if activity_preference else 0)
            Preferences.append(preferences)

        # Initialize lists to hold stability violations
        individual_stability_violations = []
        core_stability_violations = []
        individual_rationality_violations = []

        participant_names = [p.name for p in participants]
        event_names = [e.name for e in events]

        # **Individual Stability Check**
        for i in range(n):
            assigned_event_idx = assignment_dict.get(i, None)
            if assigned_event_idx is None:
                continue

            # Check Individual Stability
            for j in range(a):
                if Preferences[i][j] > Preferences[i][assigned_event_idx] and j != assigned_event_idx:
                    individual_stability_violations.append(
                        f"{participant_names[i]} can improve by switching from {event_names[assigned_event_idx]} to {event_names[j]}."
                    )

            # **Individual Rationality Check**
            if Preferences[i][assigned_event_idx] <= 0:
                individual_rationality_violations.append(
                    f"{participant_names[i]} is not individually rational in their assigned {event_names[assigned_event_idx]}."
                )

        # **Core Stability Check**
        # Use the updated core stability algorithm to check for possible reallocations
        R_values = compute_R_values_max(assignment_dict, min_bounds, max_bounds)  # Compute R values for core stability check
        for i in range(n):
            assigned_event_idx = assignment_dict.get(i, None)
            if assigned_event_idx is None:
                continue

            # Find activities where the participant would prefer to switch
            for j in range(a):
                if Preferences[i][j] > Preferences[i][assigned_event_idx] and j != assigned_event_idx:
                    # Check if a coalition of participants can move to make this switch possible
                    if algorithm_mcssp_max(R_values, max_bounds[j]) > 0:
                        core_stability_violations.append(
                            f"{participant_names[i]} and others can jointly benefit by switching to {event_names[j]}."
                        )

        # **Messages for Stability Violations**

        # Display messages for individual stability violations
        if individual_stability_violations:
            for violation in individual_stability_violations:
                messages.warning(request, violation)
            messages.error(request, "The assignment is not individually stable.")
        else:
            messages.success(request, "The assignment is individually stable.")

        # Display messages for core stability violations
        if core_stability_violations:
            for violation in core_stability_violations:
                messages.warning(request, violation)
            messages.error(request, "The assignment is not core stable.")
        else:
            messages.success(request, "The assignment is core stable.")

        # Display messages for individual rationality violations
        if individual_rationality_violations:
            for violation in individual_rationality_violations:
                messages.warning(request, violation)
            messages.error(request, "The assignment is not individually rational.")
        else:
            messages.success(request, "The assignment is individually rational.")

        # Render the results to the template
        return render(request, 'Organizer/max_allocation.html', {
            'participants': participants,
            'individual_stability_violations': individual_stability_violations,
            'core_stability_violations': core_stability_violations,
            'individual_rationality_violations': individual_rationality_violations,
        })

    except Exception as e:
        print(e)
        messages.error(request, 'Error viewing allocations!')
        return render(request, 'Organizer/max_allocation.html')


def edit_allocation_max(request):
    event = Event.objects.filter(is_active=True, created_by=request.user)
    participants = Participant.objects.filter(
        participantactivity__event__in=event).distinct()
    
    new_allocations = {}  # Dictionary to store new allocations
    activity_counts = {activity.id: 0 for activity in event}  # Track how many participants are assigned to each event

    if request.method == 'POST':
        # Process the form submission for new allocations
        for participant in participants:
            activity_id = request.POST.get(f'activity_{participant.id}')
            if activity_id:
                activity_id = int(activity_id)
                if activity_id in new_allocations:
                    new_allocations[activity_id].append(participant)
                else:
                    new_allocations[activity_id] = [participant]
                activity_counts[activity_id] += 1

        # Check for capacity issues
        for activity_id, count in activity_counts.items():
            activity = Event.objects.get(id=activity_id)
            if count > activity.max_participants:
                messages.warning(request, f"Activity '{activity.name}' capacity exceeded. Maximum capacity is {activity.max_participants}. Currently allocated: {count}.")

        # Update the participants' activity assignments
        for participant in participants:
            activity_id = request.POST.get(f'activity_{participant.id}')
            if activity_id:
                activity = Event.objects.get(id=int(activity_id))
                participant.assigned_to_max = activity
            else:
                participant.assigned_to_max = None
            participant.save()

        user = request.user
        user.is_updated_max = True
        user.save()

        messages.success(request, "Allocation updated successfully.")
        return redirect('view_allocation_max')

    # Individual Stability Check (before saving any updates)
    individual_stability_violations = []
    for participant in participants:
        current_event = participant.assigned_to_max
        participant_preferences = ParticipantActivity.objects.filter(participant=participant)

        # Get the current preferred event (if any)
        preferred_event = max(participant_preferences, key=lambda x: x.preference, default=None)

        # Check if switching would result in a better preference
        if preferred_event and current_event != preferred_event.event:
            if preferred_event.preference > participant_preferences.filter(event=current_event).first().preference:
                individual_stability_violations.append(
                    f"Participant {participant.name} can improve by switching from {current_event.name if current_event else 'None'} to {preferred_event.event.name}."
                )

    # Display warnings if individual stability is violated
    if individual_stability_violations:
        for violation in individual_stability_violations:
            messages.warning(request, violation)

    context = {
        'event': event,
        'activities': event,
        'participants': participants,
    }
    return render(request, 'Organizer/modify_allocation_max.html', context)

