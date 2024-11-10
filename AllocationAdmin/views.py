from AllocationAdmin.models import Event, ParticipantActivity, Participant
from itertools import product
from django.contrib import messages
from django.shortcuts import render, redirect, get_object_or_404
from django.db import transaction
from django.contrib.auth.decorators import login_required
# from gurobipy import Model, GRB, quicksum
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, LpInteger, LpStatus, value
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

from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, LpInteger, LpStatus, value

def solve_activity_assignment(n, a, min_bounds, max_bounds, Preferences, participant_names, event_names, initial_assignments):
    # Define the ILP model
    model = LpProblem("ActivityAssignment", LpMaximize)

    # Decision variables
    x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)}  # Participant stays in current activity
    y = {i: LpVariable(f"y_{i}", cat=LpBinary) for i in range(n)}  # Participant moves to a preferred activity
    s = {j: LpVariable(f"s_{j}", lowBound=0, cat=LpInteger) for j in range(a)}  # R-set selections

    # Objective: Maximize the sum of preferences (negative preferences will reduce the score naturally)
    model += lpSum(Preferences[i][j] * y[i] for i in range(n) for j in range(a))

    # Capacity constraints for each activity
    for j in range(a):
        min_j, max_j = min_bounds[j], max_bounds[j]
        model += lpSum(y[i] for i in range(n) if Preferences[i][j] > float('-inf')) <= max_j, f"MaxCapacity_{j}"
        model += lpSum(y[i] for i in range(n) if Preferences[i][j] > float('-inf')) >= min_j, f"MinCapacity_{j}"

    # Assignment constraint for each participant: only one state (x, y) allowed
    for i in range(n):
        model += x[i] + y[i] <= 1, f"Assignment_Constraint_{i}"

    # Calculate R-sets for coalition feasibility
    R_sets = {}
    for j in range(a):
        R_sets[j] = set([0])  # Start with {0}, no participants moving

        # Loop through possible moves to determine feasible values for R
        for h in range(1, len([i for i in range(n) if initial_assignments[i] == event_names[j]]) + 1):
            # Check if moving h participants still meets the capacity constraints for each activity
            if min_bounds[j] <= len([i for i in range(n) if initial_assignments[i] == event_names[j]]) - h <= max_bounds[j]:
                R_sets[j].add(h)

    # Convert R_sets to lists for ease of use
    R_sets = {key: sorted(values) for key, values in R_sets.items()}

    # Implement coalition feasibility by linking R-sets to actual moves
    for j in range(a):
        if len(R_sets[j]) > 1:
            model += s[j] <= max(R_sets[j]), f"R_Set_Constraint_Max_{j}"
            model += s[j] >= min(R_sets[j]), f"R_Set_Constraint_Min_{j}"

    # Solve the ILP model
    model.solve()

    # Check if the solution is optimal
    if LpStatus[model.status] != "Optimal":
        # No feasible solution was found
        return None, None  # Or handle the error as appropriate

    # Process and return results, only if s[j] has a valid value
    assignments = []
    for i in range(n):
        for j in range(a):
            if value(y[i]) == 1 and Preferences[i][j] > float('-inf'):  # Check if assigned to a preferred activity
                assignments.append((participant_names[i], event_names[j]))

    # Filter assigned activities based on valid s values
    assigned_activities = [event_names[j] for j in range(a) if value(s[j]) is not None and value(s[j]) > 0]

    return assignments, assigned_activities



# Function to compute feasible R values for each activity
# def compute_R_values(assignments, min_bounds, max_bounds):
#     """
#     Compute the R values for each activity.
#     R represents the range of feasible numbers of participants that can be moved from each activity,
#     considering both minimum and maximum bounds.
#     """
#     R = {}
#     for activity in set(assignments.values()):
#         current_count = len([k for k in assignments.values() if k == activity])
#         min_count = min_bounds[activity]
#         max_count = max_bounds[activity]

#         # Generate R values considering both lower and upper bounds
#         R[activity] = [0]
#         for h in range(1, current_count + 1):
#             if (current_count - h) >= min_count and (current_count - h) <= max_count:
#                 R[activity].append(h)
    
#     return R

# def find_b_Wbar_r(R, c):
#     """
#     Calculate breakpoint 'b', total weight 'Wbar', and maximum weight 'r' using R values.
#     Parameters:
#     R: Dictionary where keys are activity IDs and values are lists representing feasible participant moves.
#     c: Capacity for the target activity.
#     """
#     activities = list(R.keys())  # Get the list of activity IDs
#     k = len(activities)
#     b = 1
#     Wbar = 0

#     while b <= k:
#         # Get activities up to the b-th activity
#         beta_activities = activities[:b]
#         alpha_activities = activities[b:]

#         # Calculate sums for beta and alpha
#         beta_sum = sum(max(R[act]) for act in beta_activities)  # Sum of maximum values for the first 'b' activities
#         alpha_sum = sum(min(R[act]) for act in alpha_activities)  # Sum of minimum values for remaining activities
#         total_sum = beta_sum + alpha_sum

#         # Check if total sum exceeds the capacity c
#         if total_sum > c:
#             break
#         Wbar = beta_sum  # Update Wbar if the sum doesn't exceed capacity
#         b += 1

#     r = max(max(R[act]) for act in activities)  # Calculate the maximum value across all R values
#     return b, Wbar, r


# def algorithm_mcssp(R, c):
#     """Implement MCSSP using dynamic programming to determine feasible reassignments using R values."""
#     k = len(R)
#     b, Wbar, r = find_b_Wbar_r(R, c)

#     S = [[0] * (c + r + 1) for _ in range(k + 1)]
#     for participant_count in range(c - r + 1, c):
#         S[b - 1][participant_count] = 0
#     for participant_count in range(c + 1, c + r + 1):
#         S[b - 1][participant_count] = 1
#     S[b - 1][Wbar] = b

#     for t in range(b, k + 1):
#         for participant_count in range(c - r + 1, c + r + 1):
#             S[t][participant_count] = S[t - 1][participant_count]

#         max_weight = max(R[t - 1])
#         for participant_count in range(c - r + 1, c + 1):
#             for i in R[t - 1]:
#                 new_count = participant_count + i - min(R[t - 1])
#                 if 0 <= new_count < len(S[t]):
#                     S[t][new_count] = max(S[t][new_count], S[t - 1][participant_count])

#         for participant_count in range(c + max_weight, c, -1):
#             for j in range(S[t - 1][participant_count], S[t][participant_count]):
#                 for i in R[j - 1]:
#                     new_count = participant_count + i - max(R[j - 1])
#                     if 0 <= new_count < len(S[t]):
#                         S[t][new_count] = max(S[t][new_count], j)

#     column_C_values = [S[row][c] for row in range(b - 1, k + 1)]
#     optimal_solution = max(column_C_values)
#     return optimal_solution


@login_required
def allocate_participants_to_activities(request):
    events = Event.objects.filter(is_active=True, created_by=request.user)
    participants = Participant.objects.filter(participantactivity__event__in=events).distinct()

    n = participants.count()
    a = events.count()

    if n == 0 or a == 0:
        messages.warning(request, "No participants or events available for allocation.")
        return redirect('view_allocation')

    # Prepare data for ILP model
    min_bounds = list(events.values_list('min_participants', flat=True))
    max_bounds = list(events.values_list('max_participants', flat=True))
    Preferences = []
    participant_names = [p.name for p in participants]
    event_names = [e.name for e in events]
    initial_assignments = [participant.assigned_to.name for participant in participants]

    for participant in participants:
        preferences = [
            ParticipantActivity.objects.filter(participant=participant, event=event).first().preference or 0
            for event in events
        ]
        Preferences.append(preferences)

    # Solve the core stability problem with ILP
    assignments, assigned_activities = solve_activity_assignment(
        n, a, min_bounds, max_bounds, Preferences, participant_names, event_names, initial_assignments
    )

    if assignments is None:
        messages.error(request, "No feasible core stable solution found.")
        return redirect('view_allocation')

    # Update the database with the new assignments
    with transaction.atomic():
        for participant_name, event_name in assignments:
            participant = participants.get(name=participant_name)
            event = events.get(name=event_name)
            participant.assigned_to = event
            participant.save()

    messages.success(request, "Core stable allocation completed successfully.")
    return redirect('view_allocation')

@login_required
def view_allocation(request):
    try:
        # Get events and participants
        events = Event.objects.filter(created_by=request.user)
        participants = Participant.objects.filter(participantactivity__event__in=events).distinct()

        # Prepare event and participant data
        event_names = [event.name for event in events]
        participant_names = [participant.name for participant in participants]
        min_bounds = list(events.values_list('min_participants', flat=True))
        max_bounds = list(events.values_list('max_participants', flat=True))

        # Load participant preferences
        Preferences = []
        for participant in participants:
            preferences = [
                ParticipantActivity.objects.filter(participant=participant, event=event).first().preference or 0
                for event in events
            ]
            Preferences.append(preferences)

        # Perform core stability checks using the ILP model
        assignments, assigned_activities = solve_activity_assignment(
            len(participants), len(events), min_bounds, max_bounds, Preferences, participant_names, event_names, initial_assignments=[p.assigned_to.name for p in participants]
        )

        # Initialize violation lists
        individual_stability_violations = []
        individual_rationality_violations = []
        core_stability_violations = []

        # Verify individual stability and rationality
        for i, participant in enumerate(participants):
            assigned_event = assignments[i][1] if assignments and i < len(assignments) else None
            if assigned_event:
                assigned_event_idx = event_names.index(assigned_event)
                for j, event_name in enumerate(event_names):
                    if Preferences[i][j] > Preferences[i][assigned_event_idx] and event_name != assigned_event:
                        individual_stability_violations.append(
                            f"{participant_names[i]} can improve by switching from {assigned_event} to {event_name}."
                        )
                if Preferences[i][assigned_event_idx] <= 0:
                    individual_rationality_violations.append(
                        f"{participant_names[i]} is not individually rational in their assigned {assigned_event}."
                    )

        # Core stability verification
        for i in range(len(participants)):
            assigned_event = assignments[i][1] if assignments and i < len(assignments) else None
            for j in range(len(events)):
                if Preferences[i][j] > Preferences[i][event_names.index(assigned_event)] and event_names[j] != assigned_event:
                    core_stability_violations.append(
                        f"{participant_names[i]} and others can jointly benefit by switching to {event_names[j]}."
                    )

        # Display stability results
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
    for activity, participants in assignments.items():
        current_count = len(participants)
        min_count = min_bounds[activity]
        max_count = max_bounds[activity]
        
        # Initialize R[activity] with 0 (no participants leaving)
        R[activity] = {0}
        
        # Check how many participants can leave without violating min_count
        for h in range(1, current_count + 1):
            if min_count <= (current_count - h) <= max_count:
                R[activity].add(h)
    
    return R

def find_b_Wbar_r_new(R, capacity):
    """
    Calculate breakpoint 'b', total weight 'Wbar', and maximum weight 'r' using R values.
    """
    activities = list(R.keys())  # Activities involved
    k = len(activities)
    b = 1
    Wbar = 0

    while b <= k:
        # Split activities into two groups: those up to the breakpoint (b) and those after
        beta_activities = activities[:b]  # Up to b (exclusive)
        alpha_activities = activities[b:]  # From b onwards

        # Calculate sums for beta and alpha groups
        beta_sum = sum(max(R[activity]) for activity in beta_activities)
        alpha_sum = sum(min(R[activity]) for activity in alpha_activities)
        total_sum = beta_sum + alpha_sum

        # Stop if the sum exceeds the capacity
        if total_sum > capacity:
            break
        Wbar = beta_sum  # Update Wbar if the sum does not exceed capacity
        b += 1

    r = max(max(R[activity]) for activity in activities)  # Max R-value across all activities
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

    # Objective Function: Sum of non-negative preferences for maximization
    model += lpSum(max(0, Preferences[i][j]) * x[i, j] for i in range(n) for j in range(a))

    # Constraints
    # Each participant can be assigned to at most one activity
    for i in range(n):
        model += lpSum(x[i, j] for j in range(a)) <= 1, f"Participant_{i}_Assignment"

    # Capacity constraints for each activity
    for j in range(a):
        model += min_bounds[j] * y[j] <= lpSum(x[i, j] for i in range(n)), f"Min_Participants_Activity_{j}"
        model += lpSum(x[i, j] for i in range(n)) <= max_bounds[j] * y[j], f"Max_Participants_Activity_{j}"

    # Ensure each activity is activated only if assigned participants meet the bounds
    for j in range(a):
        model += lpSum(x[i, j] for i in range(n)) >= y[j], f"Activity_{j}_Activation"

    # Exclude all negative preferences from the assignment
    for i in range(n):
        for j in range(a):
            if Preferences[i][j] < 0:
                model += x[i, j] == 0, f"Exclude_Negative_Preference_{i}_{j}"

    # Solve the model
    model.solve()

    # Collect assignments and assigned activities
    assignments = [(i, j) for i in range(n) for j in range(a) if x[i, j].varValue > 0.5]
    assigned_activities = [j for j in range(a) if y[j].varValue > 0.5]

    # Mapping indices to names
    participant_names = [p.name for p in participants]
    event_names = [e.name for e in events]

    return assignments, assigned_activities, participant_names, event_names


# Updated View Allocation
@login_required
def allocate_participants_new(request):
    # Retrieve active events created by the current user (organizer)
    events = Event.objects.filter(is_active=True, created_by=request.user)
    participants = Participant.objects.filter(participantactivity__event__in=events).distinct()

    n = participants.count()
    a = events.count()

    if n == 0 or a == 0:
        messages.warning(request, "No participants or events available for allocation.")
        return redirect('view_allocation_new')

    # Get min and max bounds for events
    min_bounds = list(events.values_list('min_participants', flat=True))
    max_bounds = list(events.values_list('max_participants', flat=True))

    # Prepare Preferences matrix and collect participant/event names
    Preferences = []
    participant_names = [p.name for p in participants]
    event_names = list(events.values_list('name', flat=True))

    for participant in participants:
        preferences = []
        for event in events:
            # Fetch the preference value; if it doesn't exist, default to 0
            activity_preference = ParticipantActivity.objects.filter(participant=participant, event=event).first()
            preferences.append(activity_preference.preference if activity_preference else 0)
        Preferences.append(preferences)

    # Solve the assignment problem with core stability and negative preference exclusion
    assignments, assigned_activities, _, _ = solve_activity_assignment_pulp(
        n, a, min_bounds, max_bounds, Preferences, participants, events
    )

    # Update participant assignments in the database
    with transaction.atomic():
        for participant_idx, event_idx in assignments:
            participant = participants[participant_idx]
            event = events[event_idx]
            participant.assigned_to_new = event  # Save new assignment
            participant.save()

    # Success message for assignment completion
    messages.success(request, "Core stable assignment has been successfully calculated and saved.")
    
    return redirect('view_allocation_new')


@login_required
def view_allocation_new(request):
    try:
        # Get the events created by the current user (organizer)
        events = Event.objects.filter(created_by=request.user)
        participants = Participant.objects.filter(participantactivity__event__in=events).distinct()

        n = participants.count()
        a = events.count()

        if n == 0 or a == 0:
            messages.warning(request, "No participants or events available for viewing.")
            return redirect('home')

        # Get min and max bounds for events
        min_bounds = list(events.values_list('min_participants', flat=True))
        max_bounds = list(events.values_list('max_participants', flat=True))

        # Prepare Preferences matrix
        Preferences = []
        for participant in participants:
            preferences = [
                ParticipantActivity.objects.filter(participant=participant, event=event).first().preference or 0
                for event in events
            ]
            Preferences.append(preferences)

        # Solve the assignment problem with the core stability check
        assignments, assigned_activities, participant_names, event_names = solve_activity_assignment_pulp(
            n, a, min_bounds, max_bounds, Preferences, participants, events
        )

        # Core Stability Check
        individual_stability_violations = []
        core_stability_violations = []
        individual_rationality_violations = []

        # Validate each participant's assignment
        for i, participant in enumerate(participants):
            assigned_event_idx = next((j for j in range(a) if (i, j) in assignments), None)
            if assigned_event_idx is not None:
                assigned_event = events[assigned_event_idx]

                # Check if participant has a higher-preference event available within capacity constraints
                for j in range(a):
                    if Preferences[i][j] > Preferences[i][assigned_event_idx] and Preferences[i][j] > 0:
                        # Validate coalition feasibility for participants interested in moving to event j
                        possible_movers = [
                            p for p in range(n) if Preferences[p][j] > 0 and (p, j) not in assignments
                        ]
                        if len(possible_movers) <= max_bounds[j]:
                            core_stability_violations.append(
                                f"{participant_names[i]} and others can improve by switching to {event_names[j]}"
                            )

                # Check Individual Rationality
                if Preferences[i][assigned_event_idx] <= 0:
                    individual_rationality_violations.append(
                        f"{participant_names[i]} is not individually rational in {assigned_event.name}"
                    )

        # Process violation messages
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
    x = LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(a)), cat=LpBinary)  # x[i][j] = 1 if participant i is assigned to activity j
    y = LpVariable.dicts("y", (j for j in range(a)), cat=LpBinary)  # y[j] = 1 if activity j is assigned

    # Objective function: Maximize total preference sum, ignoring negative preferences
    prob += lpSum(max(0, Preferences[i][j]) * x[(i, j)] for i in range(n) for j in range(a)), "TotalPreferenceSum"

    # Constraints: Each participant can be assigned to at most one activity
    for i in range(n):
        prob += lpSum(x[(i, j)] for j in range(a)) <= 1, f"Participant_{i}_Assignment"

    # Capacity constraints for each activity
    for j in range(a):
        prob += min_bounds[j] * y[j] <= lpSum(x[(i, j)] for i in range(n)), f"Min_Participants_Activity_{j}"
        prob += lpSum(x[(i, j)] for i in range(n)) <= max_bounds[j] * y[j], f"Max_Participants_Activity_{j}"

    # Ensure each activity is activated only if assigned participants meet the bounds
    for j in range(a):
        prob += lpSum(x[(i, j)] for i in range(n)) >= y[j], f"Activity_{j}_Activation"

    # Solve the problem
    prob.solve()

    # Collect assignments and assigned activities
    assignments = [(i, j) for i in range(n) for j in range(a) if value(x[(i, j)]) > 0.5]
    assigned_activities = [j for j in range(a) if value(y[j]) > 0.5]

    return assignments, assigned_activities

@login_required
def allocate_activities_max(request):
    # try:
        # Get events and participants
        events = Event.objects.filter(created_by=request.user)
        participants = Participant.objects.filter(participantactivity__event__in=events).distinct()

        n = participants.count()
        a = events.count()

        if n == 0 or a == 0:
            messages.warning(request, "No participants or events available for allocation.")
            return redirect('view_allocation_max')

        min_bounds = list(events.values_list('min_participants', flat=True))
        max_bounds = list(events.values_list('max_participants', flat=True))

        Preferences = []
        for participant in participants:
            preferences = [
                ParticipantActivity.objects.filter(participant=participant, event=event).first().preference or 0
                for event in events
            ]
            Preferences.append(preferences)

        # Solve the assignment problem
        assignments, assigned_activities = solve_activity_assignment_max(
            n, a, min_bounds, max_bounds, Preferences, participants, events
        )

        # Update participant assignments in the database
        with transaction.atomic():
            for participant_idx, event_idx in assignments:
                participant = participants[participant_idx]
                event = events[event_idx]
                participant.assigned_to_max = event
                participant.save()

        messages.success(request, "Core stable assignment has been successfully calculated and saved.")
        return redirect('view_allocation_max')

    # except Exception as e:
    #     print(e)
    #     messages.error(request, 'Error during the allocation process!')
    #     return redirect('view_allocation_max')


@login_required
def view_allocation_max(request):
    try:
        # Get events and participants
        events = Event.objects.filter(created_by=request.user)
        participants = Participant.objects.filter(participantactivity__event__in=events).distinct()

        n = participants.count()
        a = events.count()

        if n == 0 or a == 0:
            messages.warning(request, "No participants or events available for viewing.")
            return redirect('home')

        min_bounds = list(events.values_list('min_participants', flat=True))
        max_bounds = list(events.values_list('max_participants', flat=True))

        Preferences = []
        assignment_dict = {}
        assignments = []  # Store current assignments

        for idx, participant in enumerate(participants):
            preferences = [
                ParticipantActivity.objects.filter(participant=participant, event=event).first().preference or 0
                for event in events
            ]
            Preferences.append(preferences)

            assigned_event = participant.assigned_to_max
            assigned_event_idx = list(events).index(assigned_event) if assigned_event in events else None
            if assigned_event_idx is not None:
                assignment_dict[idx] = assigned_event_idx
                assignments.append((idx, assigned_event_idx))

        # Core stability checks
        individual_stability_violations = []
        core_stability_violations = []
        individual_rationality_violations = []

        for i in range(n):
            assigned_event_idx = assignment_dict.get(i)
            if assigned_event_idx is None:
                continue

            # Check Individual Stability
            for j in range(a):
                if Preferences[i][j] > Preferences[i][assigned_event_idx] and Preferences[i][j] > 0:
                    # R-Set calculation for coalition feasibility
                    Ra = {0}  # Start with no participants moving
                    for h in range(1, min_bounds[assigned_event_idx]):
                        if min_bounds[assigned_event_idx] <= len([k for k in assignment_dict if assignment_dict[k] == assigned_event_idx]) - h <= max_bounds[assigned_event_idx]:
                            Ra.add(h)

                    if len(Ra) > 1 and (len([p for p in assignment_dict if Preferences[p][j] > 0]) + 1 <= max_bounds[j]):
                        core_stability_violations.append(
                            f"{participants[i].name} and others can benefit by switching to {events[j].name}."
                        )

            # Individual Rationality Check
            if Preferences[i][assigned_event_idx] <= 0:
                individual_rationality_violations.append(
                    f"{participants[i].name} is not individually rational in {events[assigned_event_idx].name}."
                )

        # Display messages based on stability checks
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

