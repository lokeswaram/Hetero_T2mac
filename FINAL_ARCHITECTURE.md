# Final Architecture: Heterogeneous T2MAC (H-T2MAC)

This document describes the complete architecture of H-T2MAC, detailing how the data, actor networks, critics, and buffers align for heterogeneous multi-agent reinforcement learning.

---

## 1. Actor Network Architecture
Each agent $i$ has a role assignment $r_i \in \{0, \dots, R-1\}$ (mapped from its SMAC unit type).

### Forward Pass (Internal State Update)
At step $t$, the agent receives observation $o_i^t$ and hidden state $h_i^{t-1}$:
1. **Role-Specific Encoder**:
   $$e_i^t = \text{EncoderBank}[r_i](o_i^t)$$
2. **Role-Specific recurrent unit**:
   $$h_i^t = \text{GRUBank}[r_i](e_i^t, h_i^{t-1})$$

### Targeted Communication & Gated Trust Gating
Each agent generates communication queries, keys, and values:
1. **Projections**:
   $$q_i^t = W_q[r_i] h_i^t, \quad k_i^t = W_k[r_i] h_i^t, \quad v_i^t = W_v[r_i] h_i^t$$
2. **Role-Aware Attention**:
   $$A_{ij}^t = \text{RoleAwareAttention}(q_i^t, k_j^t, e(r_i), e(r_j))$$
   where $e(r)$ is a learnable role embedding.
3. **Dynamic Trust Estimation**:
   $$T_{ij}^t = \text{TrustEstimator}(e(r_i), e(r_j), \text{History}_{ij}, \text{Usefulness}_i)$$
4. **Aggregation**:
   $$m_i^t = \sum_j A_{ij}^t \cdot \text{Gate}_{ij}^t \cdot T_{ij}^t \cdot v_j^t$$
5. **Combined Projection**:
   $$h_i^{\prime t} = \text{CombineProj}[r_i]([m_i^t, h_i^t])$$
6. **Policy Head**:
   $$Q_i^t = \text{PolicyHeadBank}[r_i](h_i^{\prime t})$$

---

## 2. Critic Network (Role-Aware Critic)
The `RoleAwareCritic` estimates value functions $Q(s, a)$ by conditioning on:
- Global state $s$
- Padded agent hidden states $h_{1:N}$ (padded to $\max(\text{hidden\_dim})$)
- Agent role designations $r_{1:N}$ (mapped to learnable embeddings)
- Received communication embeddings $m_{1:N}$
- Trust scores $T$

These features are concatenated and fed through fully connected layers to predict agent action values.

---

## 3. Replay Buffer & Data Flow
The replay buffer records the following transition data for each step:
- `obs`
- `actions`
- `reward`
- `terminated`
- `agent_role` (assigned unit roles)
- `role_embedding` (associated embedding states)
- `incoming_messages` (received communication embeddings)
- `outgoing_messages` (sent communication embeddings)
- `trust_scores` (computed trust levels)

---

## 4. Training Pipeline Flow
```
Sample Batch from Replay Buffer
         │
         ▼
Run MAC Forward Passes over batch sequence
         │
         ├─> Collect hidden states (padded)
         ├─> Collect communication embeddings
         └─> Collect trust scores
         │
         ▼
Compute COMALearner target values using RoleAwareCritic
         │
         ▼
Optimize Actor Network (Banks + Embeddings) & RoleAwareCritic
```
