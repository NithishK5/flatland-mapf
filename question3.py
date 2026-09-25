from lib_piglet.utils.tools import eprint
from typing import List, Tuple
import glob, os, sys, time, json
import heapq
from collections import deque
import random

# import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
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


# A lookup that turns a heading into the (row, column) step it takes on the grid.
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


DISTANCE_TABLE_CACHE = {}


def get_distance_to_goal(goal_cell, rail):
    """build_distance_to_goal with a memo in front of it.

    The rails never move and a target never changes, so a table is good for the
    whole episode. replan gets called hundreds of times and rebuilding these
    every time was most of my runtime.

    Args:
        goal_cell (tuple): (row, column) we want distances to.
        rail (GridTransitionMap): the track layout, only touched on a miss.

    Returns:
        dict: (cell, heading) -> timesteps to the goal.
    """
    cached_table = DISTANCE_TABLE_CACHE.get(goal_cell)
    if cached_table is not None:
        return cached_table

    fresh_table = build_distance_to_goal(goal_cell, rail)
    DISTANCE_TABLE_CACHE[goal_cell] = fresh_table
    return fresh_table


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


# Most states one search may expand before giving up.
#
# A search that succeeds is cheap - the distance table walks A* almost straight
# down the corridor. A search that FAILS is ruinous, because proving no route
# exists means exhausting every reachable state at every timestep. Six
# instances doing that were 92% of a 1512 second run.
MAX_SEARCH_EXPANSIONS = 150000

# A stranded train's route does not reappear one timestep later, so retrying it
# every replan call is nearly all waste. Every 25 keeps the recovery at ~4% of
# the cost. Without this one episode spent 347 seconds on failing retries.
STRANDED_RETRY_INTERVAL = 25

# When each stranded agent was last retried. Cleared per instance by get_path.
LAST_STRANDED_RETRY = {}


def find_safe_path(
    start_cell, start_heading, goal_cell, rail, existing_paths, max_timestep,
    start_time=0,
):
    """A* over (cell, heading, timestep), dodging every committed path.

    Cost equals the timestep, so g is baked into the state and we can never
    find a cheaper route to a state we have already seen. No cost bookkeeping.

    Args:
        start_cell (tuple): (row, column) we start on.
        start_heading (int): 0=N 1=E 2=S 3=W.
        goal_cell (tuple): (row, column) we must reach.
        rail (GridTransitionMap): the track layout.
        existing_paths (list): frozen paths we plan around.
        max_timestep (int): episode cutoff.
        start_time (int): when we set off. replan picks up mid-episode.

    Returns:
        list: (row, col) cells per timestep from start_time, or [].
    """
    distance_to_goal = get_distance_to_goal(goal_cell, rail)

    # Not in the table means the goal is not connected by rail at all.
    if (start_cell, start_heading) not in distance_to_goal:
        return []

    start_state = (start_cell, start_heading, start_time)

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

        # Searched long enough. A real route would have surfaced by now and
        # proving otherwise costs more than the answer is worth.
        if len(already_expanded) > MAX_SEARCH_EXPANSIONS:
            return []

        if current_cell == goal_cell:
            return rebuild_path(came_from, current_state)

        for next_state in get_timed_successors(
            current_cell, current_heading, current_time, rail, existing_paths
        ):
            next_cell, next_heading, next_time = next_state

            steps_left = distance_to_goal.get((next_cell, next_heading))
            if steps_left is None:
                continue

            # Earliest possible arrival with a clear run. Past the limit means
            # the branch is dead, and it is what keeps this search finite.
            if next_time + steps_left > max_timestep:
                continue

            if next_state in came_from:
                continue

            came_from[next_state] = current_state
            priority = next_time + steps_left

            push_order += 1
            heapq.heappush(frontier, (priority, push_order, next_state))

    return []


def work_out_heading_at(own_path, timestep, fallback_heading):
    """Reconstruct which way a train faces partway through its own path.

    A path is only cells, so the heading has to be inferred. Walk back to the
    last step where it actually moved - a parked train has identical entries in
    a row and none of them say anything about where it points.

    Args:
        own_path (list): cells indexed by timestep.
        timestep (int): the moment we want.
        fallback_heading (int): answer when it has never moved.

    Returns:
        int: 0=N 1=E 2=S 3=W.
    """
    step_index = min(timestep, len(own_path) - 1)

    while step_index > 0:
        moved_from = own_path[step_index - 1]
        moved_to = own_path[step_index]

        # Same cell means it waited, which tells us nothing.
        if moved_from != moved_to:
            row_change = moved_to[0] - moved_from[0]
            col_change = moved_to[1] - moved_from[1]

            for candidate_heading in range(4):
                if STEP_FOR_HEADING[candidate_heading] == (row_change, col_change):
                    return candidate_heading

        step_index -= 1

    return fallback_heading


def work_out_planning_order(agents, max_timestep):
    """Who gets first pick of the track.

    Whoever plans first gets a clear run and everyone after fits around them,
    so this is a real decision. Question 2 never gave us the choice - the
    evaluator called us one agent at a time in its own order.

    Soonest deadline first. Late arrival is charged twice in question 3.

    Args:
        agents (list): the EnvAgent objects.
        max_timestep (int): stand-in for an agent with no deadline, which puts
            it last where it belongs.

    Returns:
        list: agent ids in planning order.
    """

    def deadline_of(agent_index):
        agent_deadline = getattr(agents[agent_index], "deadline", None)
        if agent_deadline is None:
            return max_timestep
        return agent_deadline

    return sorted(range(len(agents)), key=deadline_of)


def paths_collide_from(first_path, second_path, from_timestep):
    """Do two finished paths ever conflict at or after a timestep?

    Used after a repair to find the knock-on damage. Same two collision types
    as is_move_blocked, but comparing two committed paths rather than testing
    one candidate move. Anything before from_timestep already executed fine.

    Args:
        first_path (list): cells indexed by timestep.
        second_path (list): cells indexed by timestep.
        from_timestep (int): ignore everything earlier.

    Returns:
        bool: True if they collide.
    """
    if not first_path or not second_path:
        return False

    # Past the shorter path's end that train is home and off the map.
    last_shared_timestep = min(len(first_path), len(second_path)) - 1

    for timestep in range(max(from_timestep, 0), last_shared_timestep + 1):
        if first_path[timestep] == second_path[timestep]:
            return True

        # Swap needs a previous timestep to compare against.
        if timestep >= 1:
            if (
                first_path[timestep] == second_path[timestep - 1]
                and second_path[timestep] == first_path[timestep - 1]
            ):
                return True

    return False


# This function returns the repaired paths after a disruption.
# @param agents A list of EnvAgent.
# @param rail The flatland railway GridTransitionMap
# @param current_timestep The timestep the malfunction/collision happened.
# @param existing_paths The paths from the previous get_path or replan.
# @param max_timestep The max timestep of this episode.
# @param new_malfunction_agents Ids that broke down this timestep.
# @param failed_agents Ids that could not reach their intended location.
# @return path_all One path per agent, history preserved.
def replan(
    agents: List[EnvAgent],
    rail: GridTransitionMap,
    current_timestep: int,
    existing_paths: List[Tuple],
    max_timestep: int,
    new_malfunction_agents: List[int],
    failed_agents: List[int],
):
    """Repair the plan from current_timestep on, after a breakdown.

    The past is not ours to edit - those trains are physically standing on
    those cells. Keep the history verbatim and rewrite only the future.

    A broken train is worse than an obstacle: it cannot move whatever we plan,
    so pin it for exactly as long as it is broken and route everyone round it.

    Args:
        agents (list): the EnvAgent objects.
        rail (GridTransitionMap): the track layout.
        current_timestep (int): when the disruption happened.
        existing_paths (list): the plan as it stood.
        max_timestep (int): episode cutoff.
        new_malfunction_agents (list): broke down this timestep. Not read -
            malfunction_data covers trains still broken from earlier too.
        failed_agents (list): could not execute their move.

    Returns:
        list: one path per agent, in agent-id order.
    """
    # Already fact: history, plus timesteps a broken train must sit out.
    frozen_paths = []

    # Where each train picks up again, or None if it needs nothing.
    resume_points = []

    for agent_index, agent in enumerate(agents):
        own_path = existing_paths[agent_index]

        if not own_path:
            frozen_paths.append([])
            resume_points.append(None)
            continue

        # Ask the SIMULATOR where it is, not our plan. An agent in
        # failed_agents is one whose plan said it moved and it did not, so the
        # plan is exactly what we cannot trust. Planning from a cell it is not
        # standing on fails on the first action, which lands it in
        # failed_agents again - a repair loop feeding on its own mistakes.
        standing_on = agent.position
        heading_now = agent.direction

        # position is None before it enters the map and after it is removed.
        if standing_on is None:
            last_known_index = min(current_timestep, len(own_path) - 1)
            standing_on = own_path[last_known_index]
            heading_now = work_out_heading_at(
                own_path, last_known_index, agent.initial_direction
            )

        if heading_now is None:
            heading_now = agent.initial_direction

        # Home already, so it is off the map and blocks nobody.
        if standing_on == agent.target:
            frozen_paths.append(list(own_path))
            resume_points.append(None)
            continue

        # Rebuild history so index still equals timestep, with the entry at
        # current_timestep corrected to where the train really is.
        history = list(own_path[:current_timestep])

        # A short path would leave a hole and shift every later index.
        while len(history) < current_timestep:
            history.append(history[-1] if history else standing_on)

        history.append(standing_on)

        # Read from malfunction_data, not new_malfunction_agents - that list
        # only holds breakdowns starting this timestep, and a train that broke
        # five steps ago is just as immovable.
        timesteps_stuck = 0
        if agent.malfunction_data:
            timesteps_stuck = agent.malfunction_data.get("malfunction", 0)

        # Pin it. Not a plan we chose, a fact others must route around.
        for _ in range(timesteps_stuck):
            history.append(standing_on)

        frozen_paths.append(history)
        resume_points.append((standing_on, heading_now, len(history) - 1))

    # Only trains actually in trouble get replanned. Rebuilding all of them was
    # slow and worse: it re-derived working plans in a different priority order
    # from the one that produced them, so good paths became worse ones.
    agents_to_repair = set()

    for agent_index, agent in enumerate(agents):
        if resume_points[agent_index] is None:
            continue

        # Still broken, whether it happened now or five steps ago.
        is_broken = False
        if agent.malfunction_data:
            is_broken = agent.malfunction_data.get("malfunction", 0) > 0

        if is_broken or agent_index in failed_agents:
            agents_to_repair.add(agent_index)
            continue

        # Parked by an earlier failed repair - its path just stops short. It is
        # not broken and it cannot be in failed_agents either, since a stopped
        # train has no move to fail. Nothing else would ever pick it up.
        own_committed_path = existing_paths[agent_index]
        if own_committed_path and own_committed_path[-1] != agent.target:
            last_tried_at = LAST_STRANDED_RETRY.get(agent_index)
            if (
                last_tried_at is None
                or current_timestep - last_tried_at >= STRANDED_RETRY_INTERVAL
            ):
                LAST_STRANDED_RETRY[agent_index] = current_timestep
                agents_to_repair.add(agent_index)

    # Start from what is already running. Trains being repaired are cut back to
    # their frozen prefix so nobody plans through where they are pinned.
    repaired_paths = []
    for agent_index in range(len(agents)):
        if agent_index in agents_to_repair:
            repaired_paths.append(list(frozen_paths[agent_index]))
        else:
            repaired_paths.append(list(existing_paths[agent_index]))

    # A repair can break somebody else's plan, and repairing them can break a
    # third. Chase it a few rounds, then stop.
    for repair_round in range(4):
        if not agents_to_repair:
            break

        for agent_index in work_out_planning_order(agents, max_timestep):
            if agent_index not in agents_to_repair:
                continue

            resume_cell, resume_heading, resume_time = resume_points[agent_index]

            # Hide our own frozen prefix, or we treat the record of where we
            # are standing as somebody else to avoid.
            other_paths = list(repaired_paths)
            other_paths[agent_index] = []

            remaining_path = find_safe_path(
                resume_cell,
                resume_heading,
                agents[agent_index].target,
                rail,
                other_paths,
                max_timestep,
                resume_time,
            )

            if remaining_path:
                # remaining_path[0] is the cell we stand on, which is also the
                # last frozen entry, so drop it from the prefix.
                repaired_paths[agent_index] = (
                    frozen_paths[agent_index][:resume_time] + remaining_path
                )
            else:
                # No way out, so park it - a stopped train costs its own
                # arrival, a crashing one costs everyone behind it too.
                #
                # But park it VISIBLY. Both collision checks read the end of a
                # path as "left the map", which is true for a train that
                # reached its target. A parked one has not. Padding to the
                # horizon says what is actually true, and without it other
                # agents plan straight through it - that was thousands of
                # replan calls in a 2400 timestep episode.
                parked_path = list(frozen_paths[agent_index])
                stuck_on = parked_path[-1]
                while len(parked_path) <= max_timestep:
                    parked_path.append(stuck_on)

                repaired_paths[agent_index] = parked_path

        # Work out who the repairs just broke, and go round again for them.
        agents_just_repaired = set(agents_to_repair)
        agents_to_repair = set()

        for other_index in range(len(agents)):
            if other_index in agents_just_repaired:
                continue
            if resume_points[other_index] is None:
                continue

            for changed_index in agents_just_repaired:
                if paths_collide_from(
                    repaired_paths[other_index],
                    repaired_paths[changed_index],
                    current_timestep,
                ):
                    # Cut back to the frozen prefix so the next round plans it
                    # fresh instead of patching a path we know is broken.
                    agents_to_repair.add(other_index)
                    repaired_paths[other_index] = list(frozen_paths[other_index])
                    break

    return repaired_paths


# How long one instance may spend looking for a better planning order. The
# question gets 3600 seconds and I was using about 200 of them.
REORDER_TIME_BUDGET_SECONDS = 30

# Give up once this many reshuffles in a row fail to improve anything.
MAX_ATTEMPTS_WITHOUT_IMPROVEMENT = 6

# Fixed seed so a run is repeatable - otherwise every measurement moves for two
# reasons at once and I cannot tell which change did what.
random.seed(0)


def list_stranded_agents(agents, planned_paths):
    """Trains this plan fails to get home.

    Empty path means the search found nothing and it never appears on the map.
    A path stopping short means it sets off and gets stuck. Both cost the full
    episode limit.

    Args:
        agents (list): the EnvAgent objects.
        planned_paths (list): one path per agent.

    Returns:
        list: ids of every stranded agent.
    """
    stranded = []

    for agent_index, agent in enumerate(agents):
        own_path = planned_paths[agent_index]

        if not own_path or own_path[-1] != agent.target:
            stranded.append(agent_index)

    return stranded


def score_plan(agents, planned_paths, max_timestep):
    """What the evaluator would charge us for a whole set of paths.

    Mirrors the marking scheme, not something convenient: arrival timestep,
    plus twice every timestep past the deadline, and an agent that never
    arrives costs the episode limit and no penalty at all.

    That last rule is why a rising penalty can mean things got better - a
    stranded agent is exempt, so getting it to turn up late trades a full
    episode-limit charge for a smaller one.

    Args:
        agents (list): the EnvAgent objects.
        planned_paths (list): one path per agent.
        max_timestep (int): episode limit, and what a no-show costs.

    Returns:
        int: lower is better, only meaningful against another plan for the
            same instance.
    """
    total_cost = 0

    for agent_index, agent in enumerate(agents):
        own_path = planned_paths[agent_index]

        if not own_path or own_path[-1] != agent.target:
            total_cost += max_timestep
            continue

        # Index equals timestep, so the last index is the arrival time.
        arrival_timestep = len(own_path) - 1
        total_cost += arrival_timestep

        agent_deadline = getattr(agent, "deadline", None)
        if agent_deadline is not None and arrival_timestep > agent_deadline:
            total_cost += 2 * (arrival_timestep - agent_deadline)

    return total_cost


def plan_everyone(agents, rail, max_timestep, planning_order):
    """Plan every agent once, in the given order, each dodging the ones before.

    Pulled out of get_path so it can run repeatedly with different orders. The
    order is the only thing that changes between calls and it changes the
    result a lot, because whoever plans first gets a clear run.

    Args:
        agents (list): the EnvAgent objects.
        rail (GridTransitionMap): the track layout.
        max_timestep (int): episode limit.
        planning_order (list): agent ids, in the order to plan them.

    Returns:
        list: one path per agent, back in agent-id order.
    """
    # Slots stay in agent-id order because that is how the evaluator reads our
    # answer, even though we fill them out of order.
    planned_paths = [[] for _ in agents]

    for agent_index in planning_order:
        agent = agents[agent_index]

        # Unfilled slots are empty and is_move_blocked skips those, so agents
        # later in the order are invisible to us for now.
        planned_paths[agent_index] = find_safe_path(
            agent.initial_position,
            agent.initial_direction,
            agent.target,
            rail,
            planned_paths,
            max_timestep,
        )

    return planned_paths


# This function returns a list of paths, one per agent.
# @param agents A list of EnvAgent, every train we have to route.
# @param rail The flatland railway GridTransitionMap
# @param max_timestep The max timestep of this episode.
# @return path_all A list of paths. Each path is a list of (x,y) tuples.
def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    # Per-instance state. Neither means anything across instances, and the
    # distance tables are keyed only by goal cell, so a stale entry would give
    # distances measured on rails that no longer exist.
    LAST_STRANDED_RETRY.clear()
    DISTANCE_TABLE_CACHE.clear()

    started_at = time.time()

    # Deadline-first. This was the whole strategy, and it is now the baseline
    # every reshuffle has to beat.
    best_order = work_out_planning_order(agents, max_timestep)
    best_paths = plan_everyone(agents, rail, max_timestep, best_order)
    best_score = score_plan(agents, best_paths, max_timestep)

    attempts_without_improvement = 0

    # Only reshuffle while somebody is actually stranded. Instances that get
    # everyone home are left alone - squeezing a working plan tighter makes it
    # brittle when a malfunction hits, which I measured the hard way.
    while list_stranded_agents(agents, best_paths):
        if time.time() - started_at >= REORDER_TIME_BUDGET_SECONDS:
            break
        if attempts_without_improvement >= MAX_ATTEMPTS_WITHOUT_IMPROVEMENT:
            break

        stranded = list_stranded_agents(agents, best_paths)

        # First try: promote every stranded agent to the front. A train with no
        # route at all was simply asked last.
        if attempts_without_improvement == 0:
            promoted = list(stranded)
        else:
            # That has stopped paying, so perturb - repeating it would only
            # propose the plan we already rejected.
            promote_count = random.randint(1, len(stranded))
            promoted = random.sample(stranded, promote_count)

            # Drag a few bystanders along. Sometimes the train that needs to
            # move is the one blocking, not the one stuck.
            bystander_count = min(len(agents) // 10, len(agents) - len(promoted))
            if bystander_count > 0:
                others = [i for i in best_order if i not in set(promoted)]
                promoted = promoted + random.sample(others, bystander_count)

        promoted_lookup = set(promoted)
        candidate_order = promoted + [
            agent_index for agent_index in best_order
            if agent_index not in promoted_lookup
        ]

        candidate_paths = plan_everyone(agents, rail, max_timestep, candidate_order)
        candidate_score = score_plan(agents, candidate_paths, max_timestep)

        # Strictly better only. Hill climbing, so we can settle on a local
        # best, but we can never end up worse than the baseline.
        if candidate_score < best_score:
            best_order = candidate_order
            best_paths = candidate_paths
            best_score = candidate_score
            attempts_without_improvement = 0
        else:
            attempts_without_improvement += 1

    return best_paths


#####################################################################
# Instantiate a Remote Client
# You should not modify codes below, unless you want to modify test_cases to test specific instance.
#####################################################################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path, sys.argv, replan=replan)
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
        deadline_files = [test.replace(".pkl", ".ddl") for test in test_cases]
        evaluator(
            get_path,
            test_cases,
            debug=debug,
            visualizer=visualizer,
            question_type=3,
            ddl=deadline_files,
            replan=replan,
            max_steps=max_steps,
        )
