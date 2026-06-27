# H-T2MAC: A Beginner's Guide

Welcome to the beginner's guide for **Heterogeneous Target-Triggered Multi-Agent Communication** (H-T2MAC). This guide will help you understand the core concepts using simple language.

---

## 1. What is T2MAC?
**T2MAC** is a method designed for robots or agents (like characters in a video game) to work together. Instead of everyone talking at the same time and creating noise, agents only speak to each other when they have important evidence or information to share. They use **attention** to listen to the most relevant speaker, and **uncertainty reduction** to decide if they need to send a message.

---

## 2. What is Heterogeneous T2MAC?
In the original version of T2MAC, every agent was considered **homogeneous** (meaning they were identical, like a team of identical clones). They shared the same brain size, looked at the world the same way, and had the same action capabilities.

**Heterogeneous T2MAC** (H-T2MAC) changes this. It allows agents to be different. In a battle game, you might have some units that are fast scouts, some that are heavily armored tanks, and some that are medics. H-T2MAC allows each unit type to have its own custom brain size, behavior, and policy.

---

## 3. What are Roles?
Roles correspond to the specializations of our agents. In StarCraft II, we map units to 5 distinct roles:
1.  **Scout**: Moves fast, spots targets.
2.  **Fighter**: Standard attacker (e.g., Marines, Marauders).
3.  **Medic**: Heals allies (e.g., Medivacs).
4.  **Tank**: Absorbs damage (e.g., Zealots).
5.  **Support**: Long-range or utility attacker (e.g., Stalkers).

Each role has a different brain capacity (hidden dimension size) that fits its complexity. For instance, a Fighter needs a larger brain (128 neurons) than a Scout (64 neurons).

---

## 4. What is Communication?
Communication is the process of agents sending messages to each other to make better decisions. For example, if a Fighter is being attacked, it can send a message to a Medic. The message is not just text, but a vector (a list of numbers) representing the agent's hidden thoughts and observations.

---

## 5. What is the Trust Module?
If agents communicate constantly, they can get confused by bad advice or old information. The **Trust Module** calculates a "trust score" between every pair of agents. It looks at:
*   Their roles (e.g., "Am I a Medic and is the sender a Fighter?").
*   Their communication history (e.g., "Did we talk recently?").
*   The usefulness of the information.

If the trust score is low, the receiver will ignore or damp down the sender's message.

---

## 6. What is the Attention Module?
The **Attention Module** helps an agent decide who to listen to. Each agent generates a "Query" (what it wants to know), and other agents provide "Keys" (what they know) and "Values" (their actual information). In H-T2MAC, we inject **learnable role embeddings** into this equation. This means agents learn to pay attention based on roles (e.g., "I should pay high attention to Medics when my health is low").

---

## 7. What Happens During One SMAC Episode?
1.  **Initialization**: The StarCraft II map loads, and units are placed. The environment initializes.
2.  **Role Mapping**: The framework looks at the unit types of the agents and assigns them roles (e.g., Marine -> Fighter).
3.  **Step Loop**:
    *   Agents look at their current observation (what is near them).
    *   They run their role-specific encoder to digest this observation.
    *   They decide whether to communicate and calculate trust scores.
    *   They query other agents, aggregate incoming messages, and compute Q-values (action values).
    *   They execute actions in the game.
    *   This repeats until the game ends (either victory or defeat).

---

## 8. What Happens During One Training Episode?
During training, we run SMAC episodes to collect experience:
1.  We collect transitions (observations, actions, rewards) from the game and save them in a **Replay Buffer**.
2.  We sample a batch of these experiences from the buffer.
3.  The **Critic** model evaluates how good the actions were and guides the updates.
4.  We adjust the weights of the role networks (Encoders, GRUs, and Policies) to improve performance in future games.

---

## 9. Why Were These Changes Added?
Without these changes, the agents are treated as identical. But in real-world tasks and complex games, a one-size-fits-all brain fails. By adding role-specific modules and role-aware communication:
*   **Medics** can focus on healing logic.
*   **Fighters** can focus on combat mechanics.
*   **Communication** becomes targeted and efficient.
*   Learning is faster and final task performance is higher.
