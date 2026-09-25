"""
This is the python script for question 1. In this script, you are required to implement a single agent path-finding algorithm
"""

from lib_piglet.utils.tools import eprint
import glob, os, sys
import heapq

# import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.utils.controller import (
        get_action,
        Train_Actions,
        Directions,
        check_conflict,
        path_controller,
        evaluator,
        remote_evaluator,
        VisualiserOptions,
    )
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    exit(1)

#########################
# Debugger and visualiser options
#########################

# Set debug to True for a step-by-step account of the run: what was planned,
# which agents malfunctioned or were blocked, and why each episode ended.
debug = False

# Controls the visualiser:
#   False                  -> run headless (fastest)
#   True                   -> watch the run with the default settings
#   VisualiserOptions(...) -> watch the run with your own settings, e.g.
#       VisualiserOptions(
#           delay=0.3,      # seconds to pause between timesteps; 0 runs at full speed
#           headless=False, # True: no window, serve to your browser instead
#           wait=True,      # hold the first frame until you are watching
#           port=8080,      # serve on a fixed port (headless mode)
#           cell_size=40,   # pixels per grid cell
#       )
visualizer = False

# Stop each episode after this many timesteps. None uses the test case's own
# limit.
max_steps = None

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
level = 0
test = 0


# Flatland numbers headings 0=N 1=E 2=S 3=W, and get_transitions() answers in
# that same order, so the index here IS the heading we end up facing.
STEP_FOR_HEADING = [
    (-1, 0),  # north is up the screen, so row goes down
    (0, 1),   # east
    (1, 0),   # south
    (0, -1),  # west
]


def get_successors(current_cell, current_heading, rail):
    """Every state reachable in one timestep.

    A state is a cell AND a heading, because rails only connect to the rails
    you drove in on. Same cell, different heading, different legal moves.

    Args:
        current_cell (tuple): (row, column) we are on.
        current_heading (int): 0=N 1=E 2=S 3=W.
        rail (GridTransitionMap): the track layout.

    Returns:
        list: (next_cell, next_heading) per legal move.
    """
    current_row, current_col = current_cell

    # Heading matters here, not just position. 4 flags, [N, E, S, W].
    allowed_headings = rail.get_transitions(current_row, current_col, current_heading)

    successors = []

    for candidate_heading in range(4):
        # No rail that way. This is also our boundary check for free - no track
        # runs off the edge of the map, so edge cells have no flag pointing out.
        if not allowed_headings[candidate_heading]:
            continue

        row_step, col_step = STEP_FOR_HEADING[candidate_heading]
        next_cell = (current_row + row_step, current_col + col_step)

        # A train faces the way it just travelled.
        successors.append((next_cell, candidate_heading))

    return successors


def estimate_steps_to_goal(current_cell, goal_cell):
    """Manhattan distance to the goal - A*'s guess at what is left.

    Admissible: one timestep moves exactly one cell in exactly one direction,
    so closing a row gap R and column gap C costs at least R + C. Rails only
    take options away, so the real trip is never shorter than this.

    Args:
        current_cell (tuple): (row, column) we are on.
        goal_cell (tuple): (row, column) we want.

    Returns:
        int: lower bound on timesteps left.
    """
    row_gap = abs(current_cell[0] - goal_cell[0])
    col_gap = abs(current_cell[1] - goal_cell[1])

    # No diagonals, so we cannot close both gaps at once.
    return row_gap + col_gap


def rebuild_path(came_from, final_state):
    """Follow the breadcrumbs back from the goal and flip them.

    Cheaper than copying a path at every step of the search.

    Args:
        came_from (dict): state -> the state we reached it from. Start maps
            to None, which is how we know to stop.
        final_state (tuple): the state we finished on.

    Returns:
        list: (row, col) cells, start to goal.
    """
    path = []
    state = final_state

    while state is not None:
        # Heading mattered for searching, but the evaluator only wants cells.
        cell = state[0]
        path.append(cell)
        state = came_from[state]

    path.reverse()

    return path


def find_shortest_path(start_cell, start_heading, goal_cell, rail):
    """A* over (cell, heading) states.

    Expands in order of f = g + h. h never overestimates, so the first time we
    pop the goal we know nothing shorter exists.

    Args:
        start_cell (tuple): (row, column) we start on.
        start_heading (int): 0=N 1=E 2=S 3=W.
        goal_cell (tuple): (row, column) we want.
        rail (GridTransitionMap): the track layout.

    Returns:
        list: (row, col) cells start to goal, or [] if no route.
    """
    # Heading is part of the state. Closing on the cell alone would seal off a
    # junction after visiting it one way and miss routes that need it another.
    start_state = (start_cell, start_heading)

    cheapest_cost_to = {start_state: 0}
    came_from = {start_state: None}
    already_expanded = set()

    frontier = []

    # heapq compares left to right, so this settles ties before it ever tries
    # to compare the states themselves. Also makes runs repeatable.
    push_order = 0

    heapq.heappush(
       frontier,
       (estimate_steps_to_goal(start_cell, goal_cell), push_order, start_state), 
    )

    while frontier:
        _, _, current_state = heapq.heappop(frontier)
        current_cell, current_heading = current_state

        # Stale copy - we found a better route to it after pushing this one.
        if current_state in already_expanded:
            continue
        already_expanded.add(current_state)

        # Goal test on the cell only. We do not care which way it ends up
        # facing, and with an admissible h the first pop is optimal.
        if current_cell == goal_cell:
            return rebuild_path(came_from, current_state)

        cost_to_here = cheapest_cost_to[current_state]

        for next_state in get_successors(current_cell, current_heading, rail):
            # Every move is one timestep.
            cost_to_next = cost_to_here + 1

            if next_state in already_expanded:
                continue

            # Seen it by a route at least as cheap, so leave the old one.
            if(
                next_state in cheapest_cost_to and
                cheapest_cost_to[next_state] <= cost_to_next
            ):
                continue

            cheapest_cost_to[next_state] = cost_to_next
            came_from[next_state] = current_state

            next_cell = next_state[0]
            priority = cost_to_next + estimate_steps_to_goal(next_cell, goal_cell)

            push_order += 1
            heapq.heappush(frontier, (priority, push_order, next_state))

    # Frontier drained, so there is no route.
    return []


# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param rail The flatland railway GridTransitionMap
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(
    start: tuple,
    start_direction: int,
    goal: tuple,
    rail: GridTransitionMap,
    max_timestep: int,
):
    # max_timestep is unused - the route is optimal, so if it does not fit,
    # nothing would have.
    return find_shortest_path(start, start_direction, goal, rail)


#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path, sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(
            os.path.join(script_path, "single_test_case/level*_test_*.pkl")
        )
        if test_single_instance:
            test_cases = glob.glob(
                os.path.join(
                    script_path,
                    "single_test_case/level{}_test_{}.pkl".format(level, test),
                )
            )
        test_cases.sort()
        evaluator(
            get_path,
            test_cases,
            debug=debug,
            visualizer=visualizer,
            question_type=1,
            max_steps=max_steps,
        )
