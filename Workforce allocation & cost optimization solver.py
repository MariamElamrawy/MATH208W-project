"""
MATH 208W – Group Project
TA Assignment LP/MILP Solver
==============================
Approach
--------
The objective (total cost including prep BUs) is non-linear in the raw
decision variables x[i,j] because prep BUs are charged once per position
opened, not per BU worked. Specifically, if applicant i is assigned to
course j, the cost is:

    wage[i] * (x[i,j]  +  prep[i,j])

where prep[i,j] is a fixed overhead that appears regardless of how many
BUs are worked. This makes the objective non-smooth and non-linear.

Linearization strategy (standard MILP technique)
-------------------------------------------------
We introduce binary indicator variables:

    y[i,j] in {0, 1}  =  1 if applicant i is assigned to course j, else 0

and link them to x[i,j] with a big-M constraint:

    x[i,j] <= bmax[i] * y[i,j]          (forces y=1 whenever x>0)

The objective then becomes fully linear:

    minimize  sum_i  sum_j  wage[i] * (x[i,j] + prep[i,j] * y[i,j])

This is a Mixed-Integer Linear Program (MILP) which we have solved here with PuLP
(open-source) using its bundled CBC solver.

"""

import pulp

# ── 1. DATA ────────────────────────────────────────────────────────────────────

courses = ["MATH150", "MATH254", "MACM316", "MATH260",
           "MATH338", "MATH342", "MATH426", "MATHACW", "MATHCW"]

# Appointment type for each course
ctype   = ["SEM",  "TUT",  "TUT",  "TUT",  "MRK",  "TUT",  "MRK",  "WKP",  "WKP"]

# BU bounds for each course (from DATA - courses & workshops tab)
cmin    = [1.5, 3,  9,  6,  0.5, 3,  0.5, 20, 12]
cmax    = [2.0, 3,  9,  7,  1.5, 3,  1.5, 25, 16]

apps = [f"App{i}" for i in range(1, 21)]

# Program and wage per BU (from DATA - parameters tab)
prog = ["PhD","MSc","PhD","MSc","PhD","MSc","MSc","PhD","PhD","PhD",
         "MSc","PhD","PhD","MSc","PhD","MSc","MSc","PhD","MSc","MSc"]
wages = [1662,1432,1662,1432,1662,1432,1432,1662,1662,1662,
         1432,1662,1662,1432,1662,1432,1432,1662,1432,1432]

# Non-prep BU bounds requested by each applicant (from DATA - applicants tab)
bmin  = [1, 2, 1, 5, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 4, 2]
bmax  = [2, 3, 1, 5, 5, 3, 4, 3, 4, 3, 4, 4, 3, 5, 2, 4, 3, 4, 4, 3]

# Prep BUs per applicant, per appointment type: [MATHACW, MATHCW, TUT, SEM, MRK]
# (from DATA - applicants tab, columns "MATHACW prep BUs" through "MRK prep BUs")
prep_data = [
    [0.57, 0.57, 1.17, 1.17, 0],   # App1
    [0.57, 1.17, 1.17, 1.17, 0],   # App2
    [0.67, 0.57, 1.17, 1.17, 0],   # App3
    [0.67, 0.57, 1.17, 1.17, 0],   # App4
    [0.17, 0.17, 1.17, 1.17, 0],   # App5
    [0.67, 1.17, 1.17, 1.17, 0],   # App6
    [0.67, 0.57, 1.17, 1.17, 0],   # App7
    [0.67, 0.57, 1.17, 1.17, 0],   # App8
    [0.67, 0.57, 1.17, 1.17, 0],   # App9
    [0.57, 0.57, 1.17, 1.17, 0],   # App10
    [0.57, 1.17, 1.17, 1.17, 0],   # App11
    [0.67, 1.17, 1.17, 1.17, 0],   # App12
    [0.57, 0.57, 1.17, 1.17, 0],   # App13
    [0.67, 0.57, 1.17, 1.17, 0],   # App14
    [0.82, 0.57, 1.17, 1.17, 0],   # App15
    [0.67, 1.17, 1.17, 1.17, 0],   # App16
    [0.57, 1.17, 1.17, 1.17, 0],   # App17
    [0.67, 1.17, 1.17, 1.17, 0],   # App18
    [0.67, 0.57, 1.17, 1.17, 0],   # App19
    [0.82, 0.57, 1.17, 1.17, 0],   # App20
]


def get_prep(i, j):
    """Return prep BU for applicant i assigned to course j."""
    if courses[j] == "MATHACW": return prep_data[i][0]
    if courses[j] == "MATHCW":  return prep_data[i][1]
    if ctype[j] == "TUT":       return prep_data[i][2]
    if ctype[j] == "SEM":       return prep_data[i][3]
    if ctype[j] == "MRK":       return prep_data[i][4]
    return 0


# ── 2. BUILDIng THE MILP ─────────────────────────────────────────────────────────

prob = pulp.LpProblem("TA_Assignment_Q3", pulp.LpMinimize)

n_apps = len(apps)
n_courses = len(courses)

# Decision variables
# x[i,j]: worked BUs assigned to applicant i for course j  (>= 0)
#          TUT columns must be integers (step = 1 BU)
# y[i,j]: binary indicator — 1 if applicant i is assigned to course j at all
x = {}
y = {}
for i in range(n_apps):
    for j in range(n_courses):
        var_type = 'Integer' if ctype[j] == 'TUT' else 'Continuous'
        x[i, j] = pulp.LpVariable(f"x_{apps[i]}_{courses[j]}", lowBound=0, cat=var_type)
        y[i, j] = pulp.LpVariable(f"y_{apps[i]}_{courses[j]}", cat='Binary')

# SEM and MRK: step = 0.5 — enforce using auxiliary integer variables h[i,j]
# x[i,j] = 0.5 * h[i,j], so h must be a non-negative integer
h = {}
for i in range(n_apps):
    for j in range(n_courses):
        if ctype[j] in ('SEM', 'MRK'):
            h[i, j] = pulp.LpVariable(f"h_{apps[i]}_{courses[j]}", lowBound=0, cat='Integer')
            prob += x[i, j] == 0.5 * h[i, j]

# Objective: minimize total cost = wage * (worked BUs + prep BUs per position)
prob += pulp.lpSum(
    wages[i] * (x[i, j] + get_prep(i, j) * y[i, j])
    for i in range(n_apps)
    for j in range(n_courses)
)

# ── 3. CONSTRAINTS ────────────────────────────────────────────────────────────

# C1: Each course's total BU assignment must be within [cmin, cmax]
for j in range(n_courses):
    prob += pulp.lpSum(x[i, j] for i in range(n_apps)) >= cmin[j], f"course_min_{courses[j]}"
    prob += pulp.lpSum(x[i, j] for i in range(n_apps)) <= cmax[j], f"course_max_{courses[j]}"

# C2: Each applicant's total worked BUs must be within their requested [bmin, bmax]
for i in range(n_apps):
    prob += pulp.lpSum(x[i, j] for j in range(n_courses)) >= bmin[i], f"app_min_{apps[i]}"
    prob += pulp.lpSum(x[i, j] for j in range(n_courses)) <= bmax[i], f"app_max_{apps[i]}"

# C3: Big-M linking constraint — y[i,j]=1 whenever x[i,j]>0
# Upper bound: x[i,j] can never exceed the applicant's max BU total
for i in range(n_apps):
    for j in range(n_courses):
        prob += x[i, j] <= bmax[i] * y[i, j], f"bigM_{apps[i]}_{courses[j]}"

# ── 4. SOLVE ──────────────────────────────────────────────────────────────────

print("Solving...")
solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=120)
prob.solve(solver)

print(f"Status:     {pulp.LpStatus[prob.status]}")
print(f"Total cost: ${pulp.value(prob.objective):,.2f}")

# ── 5. RESULTS ────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("ASSIGNMENT RESULTS")
print("="*65)

grand_total   = 0
total_worked  = 0
total_prep_bu = 0

for i, app in enumerate(apps):
    assigned = {}
    for j in range(n_courses):
        xval = pulp.value(x[i, j]) or 0
        if xval > 0.001:
            assigned[courses[j]] = round(xval, 2)

    if assigned:
        worked   = sum(assigned.values())
        prep_bu  = sum(get_prep(i, j)
                       for j in range(n_courses)
                       if courses[j] in assigned)
        cost     = wages[i] * (worked + prep_bu)
        grand_total   += cost
        total_worked  += worked
        total_prep_bu += prep_bu

        positions_str = ", ".join(f"{c}: {v} BU" for c, v in assigned.items())
        print(f"  {app:6s} ({prog[i]:3s})  |  {positions_str}")
        print(f"          worked={worked} BU, prep={prep_bu:.2f} BU, cost=${cost:,.2f}")

print("-"*65)
print(f"  Total worked BUs:  {total_worked:.2f}")
print(f"  Total prep BUs:    {total_prep_bu:.2f}")
print(f"  TOTAL COST:        ${grand_total:,.2f}")

print("\n" + "="*65)
print("COURSE COVERAGE CHECK")
print("="*65)
all_ok = True
for j, course in enumerate(courses):
    total_bu = sum(pulp.value(x[i, j]) or 0 for i in range(n_apps))
    status   = "OK" if cmin[j] <= round(total_bu, 6) <= cmax[j] else "VIOLATION"
    if status != "OK":
        all_ok = False
    print(f"  {course:10s} ({ctype[j]:3s})  assigned={round(total_bu,2):.2f}  "
          f"range=[{cmin[j]}, {cmax[j]}]  {status}")

print("\nAll course constraints satisfied:", all_ok)
