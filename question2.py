"""
This is the python script for question 2. In this script, you are required to implement a multi agent path-finding algorithm
"""

from platform import python_revision
from lib_piglet.utils.tools import eprint
from collections import deque
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
    eprint("Cannot load flatland modules!", e)
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
    """Every state the rails allow in one timestep.

    A state is a cell AND a heading, because rails only connect to the rails
    you drove in on.

    Args:
        current_cell (tuple): (row, column) we are on.
        current_heading (int): 0=N 1=E 2=S 3=W.
        rail (GridTransitionMap): the track layout.

    Returns:
        list: (next_cell, next_heading) per legal move.
    """
    current_row, current_col = current_cell

    # 4 flags back, in [N, E, S, W] order.
    allowed_headings = rail.get_transitions(current_row, current_col, current_heading)

    successors = []

    for candidate_heading in range(4):
        # No rail that way. Doubles as our boundary check - no track runs off
        # the edge of the map.
        if not allowed_headings[candidate_heading]:
            continue

        row_step, col_step = STEP_FOR_HEADING[candidate_heading]
        next_cell = (current_row + row_step, current_col + col_step)

        # A train faces the way it just travelled.
        successors.append((next_cell, candidate_heading))

    return successors


def estimate_steps_to_goal(current_cell, goal_cell):
    """Manhattan distance. Kept as the baseline I measured against.

    No longer used - build_distance_to_goal replaced it once Manhattan turned
    out to be far too loose on winding rail maps.

    Args:
        current_cell (tuple): (row, column) we are on.
        goal_cell (tuple): (row, column) we want.

    Returns:
        int: lower bound on timesteps left.
    """
    row_gap = abs(current_cell[0] - goal_cell[0])
    col_gap = abs(current_cell[1] - goal_cell[1])

    return row_gap + col_gap


def is_move_blocked(current_cell, next_cell, current_time, existing_paths):
    """Would this move crash into a train that is already planned?

    Two ways to collide and the obvious check only catches one:
      vertex - both on the same cell at the same timestep
      swap   - we trade places in one timestep, so neither is ever recorded on
               the other's cell, but head-on all the same

    Args:
        current_cell (tuple): (row, column) we are leaving.
        next_cell (tuple): (row, column) we want next. Pass current_cell to
            test whether waiting is safe.
        current_time (int): now. We land on current_time + 1.
        existing_paths (list): one path per committed train, indexed by
            timestep.

    Returns:
        bool: True if it would collide.
    """
    arrival_time = current_time + 1

    for other_path in existing_paths:
        # No path means that train never appears on the map.
        if not other_path:
            continue

        # Path ran out, so it reached its goal and Flatland removed it.
        if arrival_time >= len(other_path):
            continue

        # Vertex: standing on our destination when we get there.
        if other_path[arrival_time] == next_cell:
            return True

        # Swap: they move into the cell we leave as we move into theirs.
        if (
            other_path[arrival_time] == current_cell
            and other_path[current_time] == next_cell
        ):
            return True

    return False


def get_timed_successors(
    current_cell, current_heading, current_time, rail, existing_paths
):
    """Question 1's successors, plus time and the option to wait.

    Waiting is the whole point. On a single track with a train coming the
    other way there is no route around the problem, so sitting still until it
    passes is the only move that exists. It costs one timestep like any other
    move, so A* only spends one when it pays.

    Args:
        current_cell (tuple): (row, column) we are on.
        current_heading (int): 0=N 1=E 2=S 3=W.
        current_time (int): now.
        rail (GridTransitionMap): the track layout.
        existing_paths (list): committed paths to dodge.

    Returns:
        list: (cell, heading, timestep) per safe option.
    """
    arrival_time = current_time + 1

    # Waiting keeps our cell and heading - a stopped train has not turned.
    candidate_moves = [(current_cell, current_heading)]
    candidate_moves.extend(get_successors(current_cell, current_heading, rail))

    safe_successors = []

    for next_cell, next_heading in candidate_moves:
        # Rails say yes, traffic may still say no. Waiting gets checked too -
        # somebody could be about to drive onto the cell we are sitting on.
        if is_move_blocked(current_cell, next_cell, current_time, existing_paths):
            continue

        safe_successors.append((next_cell, next_heading, arrival_time))

    return safe_successors


def rebuild_path(came_from, final_state):
    """Follow the breadcrumbs back from the goal and flip them.

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
        # state[0] not unpacking, so this works for 2- and 3-tuples alike.
        cell = state[0]
        path.append(cell)
        state = came_from[state]

    path.reverse()

    return path


def build_distance_to_goal(goal_cell, rail):
    """Exact timesteps from every state to the goal, ignoring traffic.

    Manhattan can say 260 when the real rail distance is 900, and a heuristic
    that loose stops telling good states from bad - A* spreads out as a blob
    instead of driving at the goal. Across a time dimension that got me a
    server timeout, so I compute the truth once instead of guessing.

    Still admissible, since traffic only ever adds delay on top of this.

    Also a reachability test: a state missing from the table cannot reach the
    goal at any time, so we prune it instantly instead of searching to the
    horizon to find out.

    Args:
        goal_cell (tuple): (row, column) everything is measured to.
        rail (GridTransitionMap): the track layout.

    Returns:
        dict: (cell, heading) -> timesteps to the goal. Absent if unreachable.
    """
    # Zero for arriving at the goal whichever way we face - the goal test only
    # cares about the cell.
    distance_to_goal = {}
    queue = deque()

    for heading in range(4):
        arrival_state = (goal_cell, heading)
        distance_to_goal[arrival_state] = 0
        queue.append(arrival_state)

    while queue:
        current_cell, current_heading = queue.popleft()
        steps_from_here = distance_to_goal[(current_cell, current_heading)]

        # Backwards: arriving here facing current_heading means we stepped that
        # way, so undo the step to find the only cell we could have come from.
        row_step, col_step = STEP_FOR_HEADING[current_heading]
        previous_row = current_cell[0] - row_step
        previous_col = current_cell[1] - col_step

        # The one place that needs a real bounds check. Going forwards a True
        # flag promised the cell existed; here we invented the coordinate with
        # arithmetic and nothing has vouched for it.
        if not (0 <= previous_row < rail.height):
            continue
        if not (0 <= previous_col < rail.width):
            continue

        # No get_predecessors, so ask the forward question from the other side.
        for previous_heading in range(4):
            allowed_headings = rail.get_transitions(
                previous_row, previous_col, previous_heading
            )
            if not allowed_headings[current_heading]:
                continue

            previous_state = ((previous_row, previous_col), previous_heading)

            # BFS explores in cost order, so the first time we reach a state is
            # already its cheapest.
            if previous_state in distance_to_goal:
                continue

            distance_to_goal[previous_state] = steps_from_here + 1
            queue.append(previous_state)

    return distance_to_goal


def find_safe_path(
    start_cell, start_heading, goal_cell, rail, existing_paths, max_timestep
):
    """A* over (cell, heading, timestep), dodging every committed path.

    Simpler than question 1 despite the bigger space: cost equals the timestep,
    so g is baked into the state and we can never find a cheaper route to a
    state we have already seen. No cost bookkeeping at all.

    Args:
        start_cell (tuple): (row, column) we start on at time 0.
        start_heading (int): 0=N 1=E 2=S 3=W.
        goal_cell (tuple): (row, column) we must reach.
        rail (GridTransitionMap): the track layout.
        existing_paths (list): frozen paths we plan around.
        max_timestep (int): episode cutoff.

    Returns:
        list: (row, col) cells per timestep, or [] when traffic blocks
            everything - which the spec asks for rather than an error.
    """
    # One backward sweep buys an exact heuristic for the whole search.
    distance_to_goal = build_distance_to_goal(goal_cell, rail)

    # Not in the table means the goal is not connected by rail at all. No
    # amount of waiting fixes that.
    if (start_cell, start_heading) not in distance_to_goal:
        return []

    start_state = (start_cell, start_heading, 0)

    # Breadcrumbs and "have I queued this?" in one. Cost is fixed by the
    # timestep, so seeing a state twice can never be an improvement.
    came_from = {start_state: None}
    already_expanded = set()

    frontier = []
    push_order = 0

    heapq.heappush(
        frontier,
        (distance_to_goal[(start_cell, start_heading)], push_order, start_state),
    )

    while frontier:
        _, _, current_state = heapq.heappop(frontier)
        current_cell, current_heading, current_time = current_state

        if current_state in already_expanded:
            continue
        already_expanded.add(current_state)

        # Cell only - we do not care which way it faces or exactly when.
        if current_cell == goal_cell:
            return rebuild_path(came_from, current_state)

        for next_state in get_timed_successors(
            current_cell, current_heading, current_time, rail, existing_paths
        ):
            next_cell, next_heading, next_time = next_state

            # Absent from the table means it can never reach the goal.
            steps_left = distance_to_goal.get((next_cell, next_heading))
            if steps_left is None:
                continue

            # Earliest possible arrival even with a clear run. If that is past
            # the limit the branch is dead. Sharper than testing next_time
            # alone, and it is what keeps this search finite.
            if next_time + steps_left > max_timestep:
                continue

            if next_state in came_from:
                continue

            came_from[next_state] = current_state

            # next_time IS the cost so far, hence no g lookup.
            priority = next_time + steps_left

            push_order += 1
            heapq.heappush(frontier, (priority, push_order, next_state))

    # Drained inside the limit, so traffic blocks every route.
    return []


# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param rail The flatland railway GridTransitionMap
# @param agent_id The id of given agent
# @param existing_paths A list of lists of locations indicate existing paths.
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(
    start: tuple,
    start_direction: int,
    goal: tuple,
    rail: GridTransitionMap,
    agent_id: int,
    existing_paths: list,
    max_timestep: int,
):
    # agent_id is not needed - existing_paths only holds trains planned before
    # us, so there is no path of our own to filter out.
    return find_safe_path(
        start, start_direction, goal, rail, existing_paths, max_timestep
    )


#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path, sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(
            os.path.join(script_path, "multi_test_case/level*_test_*.pkl")
        )
        if test_single_instance:
            test_cases = glob.glob(
                os.path.join(
                    script_path,
                    "multi_test_case/level{}_test_{}.pkl".format(level, test),
                )
            )
        test_cases.sort()
        evaluator(
            get_path,
            test_cases,
            debug=debug,
            visualizer=visualizer,
            question_type=2,
            max_steps=max_steps,
        )
