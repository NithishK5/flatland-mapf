# Flatland Challenge: multi-agent train scheduling

Planning collision-free routes for up to 150 trains on a rail grid, where trains break down mid-episode and the planner has to repair the schedule while the simulation is running.

Built for FIT5222 Planning and Automated Reasoning at Monash. The solver code is mine. The simulator, the scaffold and the test instances are provided course material, credited at the bottom.

<!-- TODO: add a GIF here.
     Set `visualizer = VisualiserOptions(delay=0.1, headless=True, wait=False)`
     near the top of question3.py, run a level-3 or level-4 instance, and
     screen-record five seconds of it. Drop the file in docs/ and reference it:
     ![150 trains routing around each other](docs/flatland.gif)
     This is the single highest-value addition to this page. -->

---

## Result

Scored on a contest server against a fixed staff implementation, across 56 held-out instances.

| | Score |
|---|---|
| Q1 single agent | 15 / 15 |
| Q2 scheduling around fixed paths | 25 / 25 |
| Q3 full multi-agent with malfunctions | 50.22 / 60 |

On Q3 that is a final SIC of 463,546 with 2,827 of 2,832 trains delivered. SIC is the sum of individual costs, lower is better. For scale, the leading submissions came in roughly 10% ahead, about 32,000 SIC and five trains.

Q3 is the interesting one, so the rest of this is about Q3.

---

## The problem

Every cell holds at most one train. Rails are directional, so the moves available in a cell depend on the direction you entered it from. Each train has an expected arrival time, and arriving late costs a penalty.

The part that shapes everything is the scoring function:

- a train that arrives 50 steps late adds **50** to the cost
- a train that never arrives at all is charged the full episode limit, around **2,400**, and picks up no lateness penalty because it never arrived

So one stranded train is worth roughly fifty late ones. That rules out chasing optimal plans and rules in a fast per-train search plus a repair loop. Every design decision below follows from that one asymmetry.

On top of that, the evaluator injects a malfunction roughly every ten timesteps and calls back into a `replan` function. A stopped train blocks everything behind it, so the plan you started the episode with is obsolete almost immediately.

---

## Approach

**State is a cell, a heading and a timestep.** A rail only connects to the rail you arrived on, so position alone is not a state. Closing the search on the cell alone seals off a junction the first time it is touched and silently loses routes. Putting time in the state makes `g` equal the timestep, so there is no separate cost bookkeeping.

**Distance to goal comes from a table, not from arithmetic.** Manhattan distance is admissible but weak on a rail map, because rails do not run in straight lines. One backward breadth-first search from the goal over reversed transitions gives the exact rail distance from every state. Q2 dropped from 966 seconds to 124 on the same instances with identical output.

**Trains are planned one at a time, in an order that is searched.** Each train plans around the paths already committed. Prioritised planning is incomplete and sensitive to the order, so while any train is stranded the order is reshuffled and replanned inside a fixed time budget. The reorder only fires when someone is actually stranded, which turned out to matter more than expected. See the section on what did not work.

**Repair touches only the trains that are affected.** When a malfunction fires, the solver replans the blocked trains rather than the whole fleet.

**Failing searches are capped.** An A* search that cannot find a path will happily expand millions of time-expanded states and take the process down with it. Capping expansions turned an out-of-memory kill on the server into a clean failure, and cut a local run from 1,512 seconds to 160 with byte-identical output.

---

## Two bugs, 505 trains between them

The leaderboard gives you one number and it never tells you why. So the run logs count trains home, plan time and **replan calls per instance** separately. Replan calls was the measurement that cracked both of these.

**The solver trusted its own plan instead of the simulator.** During repair it read each train's position from the path the train was supposed to be following, rather than from where the train actually was. After a malfunction those two disagree. Reading position from the simulator instead brought **409 more trains home**, and it is the single largest correction in the project.

**Finished trains went invisible.** A train that had arrived was still physically sitting on its goal cell, but its recorded path had ended. Every other train's collision check therefore read that cell as free, routed through it, collided, and triggered another repair. On some instances the solver was replanning 2,300 times in a single episode where 20 would have been plenty. Padding a finished train's path so it stays occupied for the rest of the episode took replans from roughly 2,300 down to 20, and brought **96 more trains home**.

Neither is visible in the score. Both are obvious in the instrumentation.

---

## What did not work

Two changes that made the plan measurably better and the result measurably worse. These were more useful than most of the things that worked.

**Running the order search on every instance.** The reorder was gated behind "only if someone is stranded", which looked like leaving value on the table, so I ran it everywhere and guided it by how far each train was detouring. Plan quality improved on almost every instance. Final result: **21% worse**, with 41 fewer trains reaching their goals.

**Ordering trains by slack.** Plan the tightest deadlines first, where slack is the deadline minus the shortest possible arrival. Plan quality improved again. Final result: **9.3% worse**.

The reason is the same for both. A tighter plan is a plan with no slack in it, and slack is what absorbs a breakdown. With a malfunction every ten timesteps, the schedule that looks best on paper is the one that shatters first. The stranding gate I had written as a speed guard was doing robustness work by accident, and the only way to see that was to measure the finished episode rather than the plan.

---

## Where it breaks down

- **Prioritised planning is incomplete.** If an early train parks across the only corridor, no later train can recover and no amount of repair undoes it.
- **The order search is local.** It accepts a swap only when the whole score improves, so it settles into the first decent order it finds.
- **Repair is greedy and one-way.** An early malfunction on a busy instance sets the quality of everything after it.

The obvious next move is to stop searching over orders and start searching over subsets. Release a handful of conflicting trains, replan them together, keep the result if the episode improves. That is large neighbourhood search, and it attacks the incompleteness rather than working around it.

The other thing I would change is the test loop: inject malfunctions locally and score the finished episode, not the plan. Both negative results above would have shown up in minutes instead of costing two full evaluation runs.

---

## Run logs

`results/` holds the local evaluation runs behind the claims above. Each file ends with a summary row. Trains home is out of 2,832 across all local instances.

| Log | What changed | Trains home | Final SIC | Time |
|---|---|---|---|---|
| `q3_run.txt`, `q3_run2.txt`, `q3_run3.txt` | baseline, before the repair fixes | 2,729 | 740,967 | 197s |
| `q3_run6.txt` | both repair bugs fixed | 2,802 | 508,732 | 1,513s |
| `q3_run7.txt` | expansion cap on failing searches | 2,802 | 508,732 | **160s** |
| `q3_run8.txt` | best local configuration | **2,825** | **471,448** | 220s |
| `q3_run9.txt` | rejected variant, see what did not work | 2,784 | 572,105 | 733s |

Runs 6 and 7 are the pair worth looking at. Identical output, 9.4 times faster.

`q3_run4.txt`, `q3_run5.txt` and `q3_run10.txt` are interrupted runs kept for the per-instance detail.

---

## What is in this repo

```
question2.py     time-expanded search against a set of fixed paths
question3.py     full multi-agent solver, order search, and the replan callback
results/         local evaluation logs
```

Only my own solver code is published here. The simulator, the `lib_piglet` scaffold, the entry point and the test instances are course material and are not redistributed, so this repository is meant to be read rather than run. The environment itself is open source and available at [ShortestPathLab/flatland](https://github.com/ShortestPathLab/flatland).

---

## Credits

The Flatland environment comes from the Flatland Challenge, created by Swiss Federal Railways and run with AIcrowd. The assignment scaffold and the test instances were provided by the FIT5222 teaching team at Monash University (Daniel Harabor, Zhe Chen, Kevin Zheng). Algorithms used: A* from Hart, Nilsson and Raphael (1968), and prioritised planning from Silver (2005).

The solver logic in `question2.py` and `question3.py` is my own work.

---

## Note

This is university coursework, published as a record of the approach and not as a solution for anyone to reuse. If you are currently enrolled in this unit, submitting any of this as your own would be an academic integrity breach, and code similarity checks will find it.
