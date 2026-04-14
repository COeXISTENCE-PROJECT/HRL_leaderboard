# Centralized DQN with GlobalObservation

The centralized DQN approach for the AV routing task is built on two core components: a `DQN agent` (cDQN) and a `GlobalObservation` structure.

**GlobalObservation** maintains a centralized representation of the CAV fleet state, including agents origin, destination, departure time, selected route, and travel duration (for completed trips). It serves as the primary data source for the cDQN.

**Centralized DQN agent** (cDQN) acts as a fleet-level decision-maker with continuous access to the global fleet state. For each departing AV agent, it selects a route based on a snapshot of the `GlobalObservation` at the AV agent’s departure time.


## Technical details

### GlobalObservation

The core structure of the `GlobalObservation` class is a table of shape `num_agents × num_features`, where each row corresponds to an agent and columns represent features such as origin, destination, departure time, selected route, travel time (if completed), and boolean status indicators (is driving, finished, known).

For each agent, a per-agent observation is constructed as a snapshot of this table at the agent’s departure time. An additional one-hot column identifying the currently starting agent is added, resulting in a table of shape `num_agents × (num_features + 1)`, which is then flattened and used as input to the cDQN.

The table is initialized with empty values at the start of the experiment and incrementally populated as agents depart.

Future extensions may define alternative observation views (e.g., restricted to agents within a certain time window from the current agent). Also more clolumn-features can be added.


### cDQN agent
The cDQN is a reinforcement learning agent comprising a deep Q-network, replay buffer, exploration-exploitation policy, and training procedure.
The current implementation uses a single MLP-based Q-network (without a target network) and a FIFO replay buffer with uniform sampling. This simplified design maintains consistency with prior approaches (e.g. IQL). However, the architecture can be further developed in multiple directions (see: Extensions section).



## Key new features

Compared to previous approaches (IQL, IPPO, MAPPO, QMIX), this method provides the following new qualities:

- Utilizes **contextual information** from **all city areas where AVs travel** (as opposed to the previously used departure counts limited to the starting OD).
This gives the model the ability to capture more of the overall city dynamics, in particular implicitly learn about remote factors that influence congestion for a given vehicle.
- The defined `GlobalObservation` structure now **enables a multi-step decision making** setup, which was not available in the previous task formulation. This allows the global agent to account for the future evolution of the system, compared to making one-shot decisions at departure time by single AV agents. 

Note: from a practical perspective, the current `GlobalObservation` design does not require any additional infrastructure compared to the prior observation types (OD-departure count). It relies only on basic information that can be registered and shared by AV vehicles, which makes it feasible to consider in realistic setting.


## Further development

### Current limitations:

cDQN level:
- **uniform replay buffer storage and sampling** - introduces a signal bias toward first agents ($k^i$ possible observations for $i$-th agent, where $k$ is number of possible routes per OD). This leads to an unequal learning signal across AV agents.
- uniform exploration in a large state space (-> introduce biased exploration? e.g. towards routes with lower free-flow time). This is a more general issue and not specific to cDQN alone.

Environment representation level:
- OD pairs represented as integers -> no spatial meaning (->coordinate-based encoding can be considered),
- route choices represented as integers (meaningful representation: open problem -> ongoing path-clustering approach as a soulution?).




### Extensions / experiment ideas:

General:
- Multi-step scenarios: extend beyond single-step decision making.

Architecture develompent:
- Attention over `GlobalObservation` table (identify most relevant agents for a starting agent).
- Observation embeddings (whole table / single features / row-wise / mixed) -> reduce input size, improve feature/action representation.
- Fixed upper-bound table size -> enable reuse across varying fleet sizes (compatible with embeddings).

Information modelling:
- Time-windowed observations (focus on agents within a certain time window from now).
- Include future planned trips.

Different architectures (sequential models):
- RNN/LSTM-based agent (e.g. table embedding as hidden state, current agent data as input).


