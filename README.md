# TheBoard

## Concept

Simulate a team brainstorming retreat.

Given an LLM generated artifact - an implementation plan, a set of changes in the form of a git diff, a requirements doc, a product roadmap, etc - I would like to gather a group of carefully curated domain experts, lock them in a room for 8 hours, and have them take turns sharing ideas. The act of publicly and systematically sharing their ideas helps to feed the growth of new and better ideas.

The end result is a meticulously refined version of the original input that has been examined with a critical eye from multiple domains, personalities, and disciplines.

I would also like to ensure the human, corporate element of competition is underlying the event where potentially opposite (but valid) points of view collide resulting in healthy critical debate.

## Simulation Structure

1. A meeting is called and defined by a known and predetermined topic. Examples:

- "Take a rough spark of an idea or concept and bang out a formal set of requirements"
- "The UI for page XYZ is bland. Let's give it a style makeover"
- "Here's an app. Let's go ball's-out and envision an ambitious product roadmap that's targeted at maximizing revenue"

2. An output artifact (or set of artifacts) is defined. Examples:

- "A comprehensive product roadmap, broken into high-level MVP phases, each broken into itemized features"
- "A refined version of the input"
- "A set of recommendations, each with three options: easy, medium, hard."

3. A team is selected/created, either from a pool of specialized agents, or custom created to form a balanced team targeted to the goal and get full domain coverage

- When in doubt (or always, actually) model teams as analogs of the typical teams you see in the corporate world (devs, an architect, an eng manager, a PM, UX designer, creative director, etc)
- When deciding who is needed at the meeting, use real-world to guide decision. QA is usually not needed. Maybe one dev. Definitely at least one architect. Maybe a frontend architect too, if the situation calls for it.
- Agent pool is `/home/delorenj/code/DeLoDocs/AI/Agents/Generic` (plain text descriptions) but eventually a more formal agent definition (letta?) and pool.
- Any new agent created should be named and added to the existing pool.

4. The meeting starts

- Each agent (A, B, C) is given the context (X) and is prompted on it.
- From the Response, a neutral NoteTaker agent extracts a set of facts, ideas, tidbits. We'll call them Comments (C).
  - These are smaller than artifacts, in the same way I raise my hand to offer my idea, it's usually smaller in size than the throught i'm critiquing (or responding to).
    - If the 'response' is a pull request, the 'comment' is a...pull request comment
- The context is modified by appending the notetaker's notes and is delegated to another agent
  - To simulate real life, every agent would 'hear' agent A's 'notes' simulteously (in parallel), but since this is code, we can try different strategies to optimize.
    - how can we simulate 'a very excited agent putting his hand up after hearing something of note' ?
    - easiest (most expensive and thorough) is the greedy strategy where each round includes each agent's response delegated to each of the other agents. This results in Pn-1 responses, each broken into sets of comments extracted by the notetaker, where Pn is the number of agents in the agent pool.
    - Notation is hard to write but i'm trying to say:
      - AR1 = Agent A's round-1 turn
      - X = the current context
      - Cx = The set of facts, ideas (comments) the notetaker extracted during round x.
      - Cx = The Sum of {CxA, CxB, CxPn-1} - yes, i know Pn-1 is a number but not sure the formal way to write this and don't want to refactor all my equations now. You know what i mean though.
        - This reasoning for the above is to simulate accurately, the context is cumulative as we go around the room, since each agents 'hears' the previous agent's comments.
      - res(AR1,X) = Agent A's response after being prompted on the first round with the original, unaltered context
      - res(BR1, X+C1A) = Agent B's response after being prompted on the first round with the context containing the notetaker's round 1 notes
      - AQ1 is my attempt to give the above a named alias, but still mean Agent A's response in round 1. The 'X' is implicit
      - P is the set of all agents
      - Pn is the number of all agents.
      - P!A means all the agents, excluding A
      - res((P!A)R1X1) = The response to A's response from every other agent in round 1.
      - res(BR1, X+N1)
      - BQ1z = I needed a way to designate this agent's response in round 1 was a comment-response
    - At the end of each agent's change to comment, compression takes place. To keep our tokens in check, the notetaker does a few passes to whittle down the context.
      - Similar comments are merged
      - large, wordy comments (that can be, without significant data loss) are summarized
      - outliers and ones we have pretty much ruled out during conversation are dropped completely from the global context (all responses are actually kept in a subfolder for records)
    - A full round includes each agent's chance for full response followed by Pn-1 comments from the other agents, leading to a total of Pn^2 responses
      - Human in the loop strategy can have a user decide to continue to round two
      - Human can steer the context giving the chance to add even more value to subsequent rounds.
      - With a high number of agents equipped with cheap models, we can afford to do many rounds. If we ensure a robust diversity of domain amongst these agents, it's possible to squeeze better results.
      - There are many strategies to play with here, including employing one or two "leaders" using claude opus 4.5 and a cohort of cheap labor using DeepSeek 3.2 or Kimi-K2 Thinking.
      - Note that this is an analog to a company having an expensive Board of Director retreat, vs arguably an equally productive Engineering Team retreat.
