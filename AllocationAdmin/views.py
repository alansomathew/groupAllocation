from django.shortcuts import render, redirect
from AllocationAdmin.models import Event, ParticipantActivity, Participant
from django.contrib import messages
from django.shortcuts import render, redirect, get_object_or_404
from gurobipy import Model, GRB, quicksum
from django.contrib.auth.decorators import login_required


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
    particpantObj = ParticipantActivity.objects.filter(activity=event)
    return render(request, "Organizer/Status.html", {"data": particpantObj})


@login_required
def allocate_all_events(request):
    organizer = request.user
    events = Event.objects.filter(created_by=organizer, is_active=True)

    participants = Participant.objects.filter(
        participantactivity__activity__in=events
    ).distinct()

    if not participants.exists():
        return render(
            request,
            "Organizer/allocation.html",
            {"message": "No participants found with preferences for these activities."},
        )

    n = participants.count()
    a = events.count()

    participants_list = list(participants)
    events_list = list(events)
    lower_bounds = [event.min_participants for event in events_list]
    upper_bounds = [event.max_participants for event in events_list]

    # Create a new Gurobi model
    model = Model("Activity_Assignment")

    # Decision variables
    x = {}
    y = {}
    for i in range(n):
        for j in range(a):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x[{i},{j}]")
    for j in range(a):
        y[j] = model.addVar(vtype=GRB.BINARY, name=f"y[{j}]")

    # Update model to integrate new variables
    model.update()

    # Set objective
    model.setObjective(
        quicksum(x[i, j] for i in range(n) for j in range(a)), GRB.MAXIMIZE
    )

    # Constraints
    for i in range(n):
        model.addConstr(quicksum(x[i, j] for j in range(a)) == 1)

    for j in range(a):
        model.addConstr(
            quicksum(x[i, j] for i in range(n)) <= y[j] * upper_bounds[j]
        )
        model.addConstr(
            quicksum(x[i, j] for i in range(n)) >= y[j] * lower_bounds[j]
        )

    for i in range(n):
        for j in range(a):
            model.addConstr(x[i, j] <= y[j])

    # Optimize model
    model.optimize()

    # Prepare the results for rendering
    allocations = []
    if model.Status == GRB.OPTIMAL:
        ParticipantActivity.objects.filter(
            activity__in=events_list
        ).delete()  # Clear previous allocations
        for j in range(a):
            event_allocations = {"event": events_list[j], "participants": []}
            for i in range(n):
                if x[i, j].x > 0.5:
                    ParticipantActivity.objects.create(
                        participant=participants_list[i], activity=events_list[j]
                    )
                    event_allocations["participants"].append(participants_list[i])
            allocations.append(event_allocations)
        message = "Participants successfully allocated to the events."
    else:
        message = "No feasible solution found."

    return render(
        request, "Organizer/allocation.html", {"message": message, "allocations": allocations}
    )
