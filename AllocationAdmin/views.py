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

def solve_activity_assignment(n, a, min_bounds, max_bounds, Preferences, participant_names, event_names):
    # Define the model
    model = LpProblem("ActivityAssignment", LpMaximize)

    # Decision variables
    x = LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(a)), cat=LpBinary)
    y = LpVariable.dicts("y", (j for j in range(a)), cat=LpBinary)

    normalized_preferences = [
        [min(1, max(0, Preferences[i][j])) for j in range(a)] for i in range(n)]

    # Objective function
    model += lpSum(Preferences[i][j] * x[i, j]
                   for i in range(n) for j in range(a))

    # Constraints
    # Each participant must be assigned to at most one activity
    for i in range(n):
        model += lpSum(x[i, j] for j in range(a)) == 1

    # Ensure each activity has the correct number of participants within bounds
    for j in range(a):
        model += min_bounds[j] * y[j] <= lpSum(x[i, j] for i in range(n))
        model += lpSum(x[i, j] for i in range(n)) <= max_bounds[j] * y[j]
        model += y[j] <= 1

    # Solve the model
    model.solve()

    # Check if the solution is optimal
    if LpStatus[model.status] != "Optimal":
        return None, None

    # Collect assignments
    assignments = [(participant_names[i], event_names[j]) for i in range(n) for j in range(a) if x[i, j].varValue > 0.5]
    return assignments


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



    # Prepare preference matrix
    for participant in participants:
        preferences = [
            ParticipantActivity.objects.filter(participant=participant, event=event).first().preference or 0
            for event in events
        ]
        Preferences.append(preferences)

    # Solve the core stability problem with ILP
    assignments = solve_activity_assignment(
        n, a, min_bounds, max_bounds, Preferences, participant_names, event_names,
    )

    if assignments is None:
        messages.error(request, "No feasible core stable solution found.")
        return redirect('view_allocation')

    # Update the database with the new assignments
    with transaction.atomic():
        for participant_name, event_name in assignments:
            participant = participants.get(name=participant_name)
            print(participant)
            event = events.get(name=event_name)
            print(events)
            participant.assigned_to = event
            participant.save()

    messages.success(request, "Core stable allocation completed successfully.")
    return redirect('view_allocation')


@login_required
def view_allocation(request):
    try:
        # Retrieve events created by the current user and participants for these events
        events = Event.objects.filter(created_by=request.user)
        participants = Participant.objects.filter(participantactivity__event__in=events).distinct()
        n = participants.count()
        a = events.count()

        # Prepare lists of event and participant names, and min/max bounds for each event
        event_names = [event.name for event in events]
        participant_names = [participant.name for participant in participants]
        min_bounds = list(events.values_list('min_participants', flat=True))
        max_bounds = list(events.values_list('max_participants', flat=True))

        # Generate a preference matrix for each participant across all events
        Preferences = []
        for participant in participants:
            preferences = [
                ParticipantActivity.objects.filter(participant=participant, event=event).first().preference or 0
                for event in events
            ]
            Preferences.append(preferences)

        # Assume `assignments` is provided by solve_activity_assignment function
        assignments = solve_activity_assignment(
            n, a, min_bounds, max_bounds, Preferences, participant_names, event_names
        )

        # Initialize violation lists to store any individual stability, rationality, or core stability issues
        individual_stability_violations = []
        individual_rationality_violations = []
        core_stability_violations = []

        # Core Stability Check
        for i, participant in enumerate(participants):
            assigned_event_name = participant.assigned_to.name  # Current assigned event for participant
            assigned_event_idx = event_names.index(assigned_event_name)  # Index of assigned event in event_names list
            preference_assigned_event = Preferences[i][assigned_event_idx]  # Preference value for assigned event

            # Initialize a list to store possible coalitions for each alternative activity
            coalition_messages = []

            # 1. **Individual Rationality Check**
            # Check if the participant's assigned event has a non-negative preference
            if preference_assigned_event < 0:
                individual_rationality_violations.append(
                    f"{participant_names[i]} is not individually rational in {assigned_event_name} (preference {preference_assigned_event})."
                )

            # 2. **Individual Stability Check**
            # Check if there exists any event with a higher preference than the current assignment
            for j, event_name in enumerate(event_names):
                if Preferences[i][j] > preference_assigned_event:
                    individual_stability_violations.append(
                        f"{participant_names[i]} can improve by switching from {assigned_event_name} to {event_name}."
                    )
                    break  # Stop after finding the first better alternative


            # Core Stability: Check feasibility for each alternative activity `b` where preference is higher
            for j, event_name in enumerate(event_names):
                if Preferences[i][j] > preference_assigned_event:
                    target_event_idx = j  # The index for the target event `b`

                    # Construct B_set for participants who strictly prefer target event `b` over current assignment
                    B_set = [
                        k for k in range(len(participants))
                        if Preferences[k][target_event_idx] >= Preferences[k][event_names.index(participants[k].assigned_to.name)]
                    ]

                    print(f"Preferences for participants: {Preferences}")
                    print(f"B_set for {participant_names[i]} moving to {event_names[target_event_idx]}: {B_set}")

                    # Initialize a dictionary to hold R sets for each activity
                    R_sets = {}

                    # For each activity `c` other than the target event `b`
                    for c_idx, c_name in enumerate(event_names):
                        if c_name != event_name:
                            Rc = set()  # Initialize Rc with feasible move counts
                            
                            # Define participants assigned to activity `c`
                            current_c_participants = [
                                k for k in range(len(participants)) if participants[k].assigned_to.name == c_name
                            ]
                            
                            # Determine eligible participants from `c` who are in `B_set`
                            eligible_to_move_from_c = [p for p in current_c_participants if p in B_set]
                            
                            # Calculate Rc for feasible moves from `c` without violating capacity
                            for h in range(1, len(eligible_to_move_from_c) + 1):
                                remaining_capacity = len(current_c_participants) - h
                                if min_bounds[c_idx] <= remaining_capacity <= max_bounds[c_idx]:
                                    Rc.add(h)  # Add feasible move count `h`

                            print(f"Rc set for {c_name}: {Rc}")
                            R_sets[c_name] = list(Rc)  # Store Rc in the R_sets dictionary

                    # Apply ILP to check for feasible coalition from R_sets
                    prob = LpProblem("Feasibility_Check", LpMaximize)

                    # Define decision variables for each event's R set
                    h_vars = {
                        c_name: LpVariable.dicts(f"h_{c_name}", R_sets[c_name], cat="Binary")
                        for c_name in R_sets
                    }

                    # Objective function: Dummy objective to focus on feasibility
                    prob += 0, "Dummy_Objective"

                    # Constraints: Select exactly one feasible value from each R_c
                    for c_name, h_var in h_vars.items():
                        prob += lpSum(h_var[h] for h in R_sets[c_name]) == 1, f"OneValueFromR_{c_name}"

                    # Capacity constraint for target event `b` after coalition movement
                    total_move_to_b = lpSum(
                        h * h_vars[c_name][h] for c_name in R_sets for h in R_sets[c_name]
                    )
                    
                    # Calculate the current number of participants assigned to `b`
                    current_b_participants = len([
                        k for k in range(len(participants)) if participants[k].assigned_to.name == event_name
                    ])

                    # Capacity constraint for the target event
                    prob += (
                        min_bounds[target_event_idx] <= current_b_participants + total_move_to_b <= max_bounds[target_event_idx],
                        "CapacityConstraint_TargetEvent"
                    )

                    # Solve the ILP model
                    prob.solve()

                    # Check if the solution is feasible
                    if LpStatus[prob.status] == 'Optimal':
                        coalition_found = True
                        coalition_participants = [
                            participant_names[p] for p in B_set if any(h_vars[c_name][h].varValue == 1 for h in R_sets[c_name])
                        ]
                        coalition_messages.append(
                            f"{participant_names[i]} could improve by moving to {event_names[target_event_idx]} with a coalition of "
                            f"{', '.join(coalition_participants)} from other activities."
                        )
                    else:
                        coalition_messages.append(
                            f"No feasible coalition found for {participant_names[i]} to move to {event_names[target_event_idx]}."
                        )

            # Append messages to core stability violations
            core_stability_violations.extend(coalition_messages)


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
        # Retrieve events and participants created by the current user
        events = Event.objects.filter(created_by=request.user)
        participants = Participant.objects.filter(participantactivity__event__in=events).distinct()
        n = participants.count()
        a = events.count()

        event_names = [event.name for event in events]
        participant_names = [participant.name for participant in participants]
        min_bounds = list(events.values_list('min_participants', flat=True))
        max_bounds = list(events.values_list('max_participants', flat=True))

        # Generate preference matrix
        Preferences = []
        for participant in participants:
            preferences = [
                ParticipantActivity.objects.filter(participant=participant, event=event).first().preference or 0
                for event in events
            ]
            Preferences.append(preferences)

        
        # Initialize assignments dictionary to keep track of participants assigned to each event
        assignments = {event_name: [] for event_name in event_names}
        for participant in participants:
            assigned_event_name = participant.assigned_to_new.name  # Get the assigned event for each participant
            assignments[assigned_event_name].append(participant)

        print(assignments)  # Debugging output to check assignments structure

        # Initialize lists for stability violations
        individual_stability_violations = []
        individual_rationality_violations = []
        core_stability_violations = []

        # Core Stability Check
        for i, participant in enumerate(participants):
            assigned_event_name = participant.assigned_to_new.name  # Current assigned event for participant
            assigned_event_idx = event_names.index(assigned_event_name)  # Index of assigned event in event_names list
            preference_assigned_event = Preferences[i][assigned_event_idx]  # Preference value for assigned event

            # Initialize a list to store possible coalitions for each alternative activity
            coalition_messages = []

            # 1. **Individual Rationality Check**
            # Check if the participant's assigned event has a non-negative preference
            if preference_assigned_event < 0:
                individual_rationality_violations.append(
                    f"{participant_names[i]} is not individually rational in {assigned_event_name} (preference {preference_assigned_event})."
                )

            # 2. **Individual Stability Check**
            # Check if there exists any event with a higher preference than the current assignment
            for j, event_name in enumerate(event_names):
                if Preferences[i][j] > preference_assigned_event:
                    individual_stability_violations.append(
                        f"{participant_names[i]} can improve by switching from {assigned_event_name} to {event_name}."
                    )
                    break  # Stop after finding the first better alternative


            # Core Stability: Check feasibility for each alternative activity `b` where preference is higher
            for j, event_name in enumerate(event_names):
                if Preferences[i][j] > preference_assigned_event:
                    target_event_idx = j  # The index for the target event `b`

                    # Construct B_set for participants who strictly prefer target event `b` over current assignment
                    B_set = [
                        k for k in range(len(participants))
                        if Preferences[k][target_event_idx] >= Preferences[k][event_names.index(participants[k].assigned_to_new.name)]
                    ]

                    print(f"Preferences for participants: {Preferences}")
                    print(f"B_set for {participant_names[i]} moving to {event_names[target_event_idx]}: {B_set}")

                    # Initialize a dictionary to hold R sets for each activity
                    R_sets = {}

                    # For each activity `c` other than the target event `b`
                    for c_idx, c_name in enumerate(event_names):
                        if c_name != event_name:
                            Rc = set()  # Initialize Rc with feasible move counts
                            
                            # Define participants assigned to activity `c`
                            current_c_participants = [
                                k for k in range(len(participants)) if participants[k].assigned_to_new.name == c_name
                            ]
                            
                            # Determine eligible participants from `c` who are in `B_set`
                            eligible_to_move_from_c = [p for p in current_c_participants if p in B_set]
                            
                            # Calculate Rc for feasible moves from `c` without violating capacity
                            for h in range(1, len(eligible_to_move_from_c) + 1):
                                remaining_capacity = len(current_c_participants) - h
                                if min_bounds[c_idx] <= remaining_capacity <= max_bounds[c_idx]:
                                    Rc.add(h)  # Add feasible move count `h`

                            print(f"Rc set for {c_name}: {Rc}")
                            R_sets[c_name] = list(Rc)  # Store Rc in the R_sets dictionary

                    # Apply ILP to check for feasible coalition from R_sets
                    prob = LpProblem("Feasibility_Check", LpMaximize)

                    # Define decision variables for each event's R set
                    h_vars = {
                        c_name: LpVariable.dicts(f"h_{c_name}", R_sets[c_name], cat="Binary")
                        for c_name in R_sets
                    }

                    # Objective function: Dummy objective to focus on feasibility
                    prob += 0, "Dummy_Objective"

                    # Constraints: Select exactly one feasible value from each R_c
                    for c_name, h_var in h_vars.items():
                        prob += lpSum(h_var[h] for h in R_sets[c_name]) == 1, f"OneValueFromR_{c_name}"

                    # Capacity constraint for target event `b` after coalition movement
                    total_move_to_b = lpSum(
                        h * h_vars[c_name][h] for c_name in R_sets for h in R_sets[c_name]
                    )
                    
                    # Calculate the current number of participants assigned to `b`
                    current_b_participants = len([
                        k for k in range(len(participants)) if participants[k].assigned_to_new.name == event_name
                    ])

                    # Capacity constraint for the target event
                    prob += (
                        min_bounds[target_event_idx] <= current_b_participants + total_move_to_b <= max_bounds[target_event_idx],
                        "CapacityConstraint_TargetEvent"
                    )

                    # Solve the ILP model
                    prob.solve()

                    # Check if the solution is feasible
                    if LpStatus[prob.status] == 'Optimal':
                        coalition_found = True
                        coalition_participants = [
                            participant_names[p] for p in B_set if any(h_vars[c_name][h].varValue == 1 for h in R_sets[c_name])
                        ]
                        coalition_messages.append(
                            f"{participant_names[i]} could improve by moving to {event_names[target_event_idx]} with a coalition of "
                            f"{', '.join(coalition_participants)} from other activities."
                        )
                    else:
                        coalition_messages.append(
                            f"No feasible coalition found for {participant_names[i]} to move to {event_names[target_event_idx]}."
                        )

            # Append messages to core stability violations
            core_stability_violations.extend(coalition_messages)

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
    try:
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

    except Exception as e:
        print(e)
        messages.error(request, 'Error during the allocation process!')
        return redirect('view_allocation_max')


@login_required
def view_allocation_max(request):
    try:
        # Get events and participants
        # Retrieve events and participants created by the current user
        events = Event.objects.filter(created_by=request.user)
        participants = Participant.objects.filter(participantactivity__event__in=events).distinct()
        n = participants.count()
        a = events.count()

        event_names = [event.name for event in events]
        participant_names = [participant.name for participant in participants]
        min_bounds = list(events.values_list('min_participants', flat=True))
        max_bounds = list(events.values_list('max_participants', flat=True))

        # Generate preference matrix
        Preferences = []
        for participant in participants:
            preferences = [
                ParticipantActivity.objects.filter(participant=participant, event=event).first().preference or 0
                for event in events
            ]
            Preferences.append(preferences)

        
        # Initialize assignments dictionary to keep track of participants assigned to each event
        assignments = {event_name: [] for event_name in event_names}
        for participant in participants:
            assigned_event_name = participant.assigned_to_max.name  # Get the assigned event for each participant
            assignments[assigned_event_name].append(participant)

        print(assignments)  # Debugging output to check assignments structure

        # Initialize lists for stability violations
        individual_stability_violations = []
        individual_rationality_violations = []
        core_stability_violations = []

        # Core Stability Check
        for i, participant in enumerate(participants):
            assigned_event_name = participant.assigned_to_max.name  # Current assigned event for participant
            assigned_event_idx = event_names.index(assigned_event_name)  # Index of assigned event in event_names list
            preference_assigned_event = Preferences[i][assigned_event_idx]  # Preference value for assigned event

            # Initialize a list to store possible coalitions for each alternative activity
            coalition_messages = []

            # 1. **Individual Rationality Check**
            # Check if the participant's assigned event has a non-negative preference
            if preference_assigned_event < 0:
                individual_rationality_violations.append(
                    f"{participant_names[i]} is not individually rational in {assigned_event_name} (preference {preference_assigned_event})."
                )

            # 2. **Individual Stability Check**
            # Check if there exists any event with a higher preference than the current assignment
            for j, event_name in enumerate(event_names):
                if Preferences[i][j] > preference_assigned_event:
                    individual_stability_violations.append(
                        f"{participant_names[i]} can improve by switching from {assigned_event_name} to {event_name}."
                    )
                    break  # Stop after finding the first better alternative


            # Core Stability: Check feasibility for each alternative activity `b` where preference is higher
            for j, event_name in enumerate(event_names):
                if Preferences[i][j] > preference_assigned_event:
                    target_event_idx = j  # The index for the target event `b`

                    # Construct B_set for participants who strictly prefer target event `b` over current assignment
                    B_set = [
                        k for k in range(len(participants))
                        if Preferences[k][target_event_idx] >= Preferences[k][event_names.index(participants[k].assigned_to_max.name)]
                    ]

                    print(f"Preferences for participants: {Preferences}")
                    print(f"B_set for {participant_names[i]} moving to {event_names[target_event_idx]}: {B_set}")

                    # Initialize a dictionary to hold R sets for each activity
                    R_sets = {}

                    # For each activity `c` other than the target event `b`
                    for c_idx, c_name in enumerate(event_names):
                        if c_name != event_name:
                            Rc = set()  # Initialize Rc with feasible move counts
                            
                            # Define participants assigned to activity `c`
                            current_c_participants = [
                                k for k in range(len(participants)) if participants[k].assigned_to_max.name == c_name
                            ]
                            
                            # Determine eligible participants from `c` who are in `B_set`
                            eligible_to_move_from_c = [p for p in current_c_participants if p in B_set]
                            
                            # Calculate Rc for feasible moves from `c` without violating capacity
                            for h in range(1, len(eligible_to_move_from_c) + 1):
                                remaining_capacity = len(current_c_participants) - h
                                if min_bounds[c_idx] <= remaining_capacity <= max_bounds[c_idx]:
                                    Rc.add(h)  # Add feasible move count `h`

                            print(f"Rc set for {c_name}: {Rc}")
                            R_sets[c_name] = list(Rc)  # Store Rc in the R_sets dictionary

                    # Apply ILP to check for feasible coalition from R_sets
                    prob = LpProblem("Feasibility_Check", LpMaximize)

                    # Define decision variables for each event's R set
                    h_vars = {
                        c_name: LpVariable.dicts(f"h_{c_name}", R_sets[c_name], cat="Binary")
                        for c_name in R_sets
                    }

                    # Objective function: Dummy objective to focus on feasibility
                    prob += 0, "Dummy_Objective"

                    # Constraints: Select exactly one feasible value from each R_c
                    for c_name, h_var in h_vars.items():
                        prob += lpSum(h_var[h] for h in R_sets[c_name]) == 1, f"OneValueFromR_{c_name}"

                    # Capacity constraint for target event `b` after coalition movement
                    total_move_to_b = lpSum(
                        h * h_vars[c_name][h] for c_name in R_sets for h in R_sets[c_name]
                    )
                    
                    # Calculate the current number of participants assigned to `b`
                    current_b_participants = len([
                        k for k in range(len(participants)) if participants[k].assigned_to_max.name == event_name
                    ])

                    # Capacity constraint for the target event
                    prob += (
                        min_bounds[target_event_idx] <= current_b_participants + total_move_to_b <= max_bounds[target_event_idx],
                        "CapacityConstraint_TargetEvent"
                    )

                    # Solve the ILP model
                    prob.solve()

                    # Check if the solution is feasible
                    if LpStatus[prob.status] == 'Optimal':
                        coalition_found = True
                        coalition_participants = [
                            participant_names[p] for p in B_set if any(h_vars[c_name][h].varValue == 1 for h in R_sets[c_name])
                        ]
                        coalition_messages.append(
                            f"{participant_names[i]} could improve by moving to {event_names[target_event_idx]} with a coalition of "
                            f"{', '.join(coalition_participants)} from other activities."
                        )
                    else:
                        coalition_messages.append(
                            f"No feasible coalition found for {participant_names[i]} to move to {event_names[target_event_idx]}."
                        )

            # Append messages to core stability violations
            core_stability_violations.extend(coalition_messages)

        # Process violation messages
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

