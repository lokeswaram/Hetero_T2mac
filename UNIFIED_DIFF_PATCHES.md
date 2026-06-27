# Unified Diff Patches for H-T2MAC Migration

This document records the exact changes made to the existing T2MAC files to implement the heterogeneous extensions.

---

## 1. `src/modules/agents/__init__.py`
```diff
@@ -7,6 +7,7 @@
 from .tmac_full_comm_rnn_msg_agent import RnnMsgAgent as TFCMsgAgent
 from .tmac_p2p_comm_rnn_msg_agent import RnnMsgAgent as P2PMsgAgent
 from .tmac_comm_rate_rnn_msg_agent import RnnMsgAgent as CRMsgAgent
+from .heterogeneous_agent import HeterogeneousAgent
 
 REGISTRY["rnn"] = RNNAgent
 REGISTRY['rnn_msg'] = RnnMsgAgent
@@ -13,4 +13,5 @@
 REGISTRY['tmac_vffac_rnn_msg'] = RnnMsgAgent
 REGISTRY['tmac_full_comm_rnn_msg'] = TFCMsgAgent
 REGISTRY['tmac_p2p_comm_rnn_msg'] = P2PMsgAgent
-REGISTRY['tmac_comm_rate_rnn_msg'] = CRMsgAgent
+REGISTRY['tmac_comm_rate_rnn_msg'] = CRMsgAgent
+REGISTRY['heterogeneous_agent'] = HeterogeneousAgent
```

---

## 2. `src/modules/critics/__init__.py`
```diff
@@ -1,0 +1,2 @@
+from .coma import COMACritic
+from .role_aware_critic import RoleAwareCritic
```

---

## 3. `src/controllers/basic_controller.py`
```diff
@@ -8,9 +8,15 @@
 class BasicMAC:
     def __init__(self, scheme, groups, args):
         self.n_agents = args.n_agents
         self.args = args
         input_shape = self._get_input_shape(scheme)
+        
+        if getattr(args, "heterogeneous", False):
+            from smac.role_mapper import SMACRoleMapper
+            self.role_mapper = SMACRoleMapper(args.role_names)
+            self.agent_roles = None
+            
         self._build_agents(input_shape)
         self.agent_output_type = args.agent_output_type
 
@@ -14,6 +20,15 @@
 
         self.hidden_states = None
 
+    def set_roles(self, roles):
+        self.agent_roles = roles
+
+    def get_role_embeddings(self):
+        if getattr(self.args, "heterogeneous", False) and self.agent_roles is not None:
+            roles_tensor = th.tensor(self.agent_roles, device=self.args.device)
+            return self.agent.role_embeddings(roles_tensor)
+        return None
+
     def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
         # Only select actions for the selected batch elements in bs
@@ -23,7 +38,12 @@
     def forward(self, ep_batch, t, test_mode=False):
         agent_inputs = self._build_inputs(ep_batch, t)
         avail_actions = ep_batch["avail_actions"][:, t]
-        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
+        
+        if getattr(self.args, "heterogeneous", False):
+            if "agent_role" in ep_batch.data.transition_data:
+                roles = ep_batch["agent_role"][:, t].squeeze(-1)[0].long().cpu().numpy().tolist()
+            else:
+                roles = self.agent_roles
+            self.hidden_states = self.agent(agent_inputs, self.hidden_states, roles)
+            agent_outs = self.agent.q_without_communication(self.hidden_states, roles)
+        else:
+            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
 
@@ -50,3 +70,12 @@
     def init_hidden(self, batch_size):
-        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
+        if getattr(self.args, "heterogeneous", False):
+            self.hidden_states = []
+            for agent_id in range(self.n_agents):
+                role_idx = self.agent_roles[agent_id]
+                role_name = self.args.role_names[role_idx]
+                h_dim = self.args.roles[role_name].get("hidden_dim", 64)
+                h = th.zeros(batch_size, h_dim, device=self.args.device)
+                self.hidden_states.append(h)
+        else:
+            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
 
@@ -66,7 +95,11 @@
     def _build_agents(self, input_shape):
-        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
+        if getattr(self.args, "heterogeneous", False):
+            from modules.agents.heterogeneous_agent import HeterogeneousAgent
+            self.agent = HeterogeneousAgent(input_shape, self.args)
+        else:
+            self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
```

---

## 4. `src/controllers/tmac_p2p_comm_controller.py`
The modifications to `VffacMAC` match the `BasicMAC` changes, extending the `forward` pass to handle communication aggregation conditioning on trust metrics and role embeddings, and updating parameter lists for the `msg_optim`.

---

## 5. `src/learners/coma_learner.py`
The `COMALearner` collects internal target/recurrent representations via `_collect_agent_data` during training, and passes these variables into the `RoleAwareCritic` instead of the homogeneous `COMACritic`.

---

## 6. `src/run.py`
```diff
@@ -93,6 +93,14 @@
         "reward": {"vshape": (1,)},
         "terminated": {"vshape": (1,), "dtype": th.uint8},
     }
+    if getattr(args, "heterogeneous", False):
+        scheme.update({
+            "agent_role": {"vshape": (1,), "group": "agents", "dtype": th.long},
+            "role_embedding": {"vshape": (args.role_emb_dim,), "group": "agents", "dtype": th.float32},
+            "incoming_messages": {"vshape": (args.n_value,), "group": "agents", "dtype": th.float32},
+            "outgoing_messages": {"vshape": (args.n_value,), "group": "agents", "dtype": th.float32},
+            "trust_scores": {"vshape": (args.n_agents,), "group": "agents", "dtype": th.float32},
+        })
     groups = {
         "agents": args.n_agents
     }
```

---

## 7. `src/runners/episode_runner.py`
The transition and reset routines assign unit roles at environment initialization, extract intermediate communication metrics, and store them at each step in the extended buffer layout.
