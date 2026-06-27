import torch
import torch as th
import itertools
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents.heterogeneous_agent import HeterogeneousAgent
from smac_hetero.role_mapper import SMACRoleMapper

class VffacMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        
        self.role_mapper = SMACRoleMapper(args.role_names)
        self.agent_roles = None
            
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        
        param = itertools.chain(self.agent.msg_proj1.parameters(), self.agent.msg_proj2.parameters())
        self.msg_optim = torch.optim.Adam(param, lr=0.001)
        self.crit_fun = torch.nn.CrossEntropyLoss().cuda()

    def set_roles(self, roles):
        self.agent_roles = roles

    def get_role_embeddings(self):
        if self.agent_roles is not None:
            roles_tensor = torch.tensor(self.agent_roles, device=self.args.device)
            return self.agent.role_embeddings(roles_tensor)
        return None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions

    @torch.no_grad()
    def generate_send_target(self, send_prob):
        send_target = torch.where(send_prob > 0.75, 1, 0)
        if len(send_target.shape) == 2:
            send_target = send_target.unsqueeze(0)
        return send_target

    def forward(self, ep_batch, t, test_mode=False, counterfactual=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if "agent_role" in ep_batch.data.transition_data:
            roles = ep_batch["agent_role"][:, t].squeeze(-1)[0].long().cpu().numpy().tolist()
        else:
            roles = self.agent_roles

        self.hidden_states = self.agent(agent_inputs, self.hidden_states, roles)
        if counterfactual:
            q_without_comm = self.agent.q_without_communication(self.hidden_states, roles)

        agents_key, agents_value, agents_query = self.agent.communicate(self.hidden_states, roles)

        send_prob = self.agent.generate_send_prob(agent_inputs, roles)

        send_target = torch.clamp(self.generate_send_target(send_prob) - torch.eye(self.n_agents).cuda(), 0,
                                  1).int().detach()

        if not hasattr(self, "comm_history") or self.comm_history is None:
            self.comm_history = torch.zeros(ep_batch.batch_size, self.n_agents, self.n_agents, self.args.history_len, device=agent_inputs.device)
            self.past_usefulness = torch.zeros(ep_batch.batch_size, self.n_agents, self.n_agents, 1, device=agent_inputs.device)

        agents_out, cmb_u, incoming_msgs, trust_scores = self.agent.aggregate(
            agents_query, agents_key, agents_value,
            self.hidden_states, send_target, roles,
            self.comm_history, self.past_usefulness
        )
        
        # Roll history
        self.comm_history = torch.cat([self.comm_history[..., 1:], send_target.unsqueeze(-1).float()], dim=-1)
        self.past_usefulness = cmb_u.unsqueeze(2).repeat(1, 1, self.n_agents, 1)

        # Expose logged values for step update
        self.incoming_messages = incoming_msgs.mean(dim=0).detach()
        self.outgoing_messages = agents_value.mean(dim=0).detach()
        self.trust_scores = trust_scores.mean(dim=0).detach()

        # Calculate u_err for the sender loss
        ori_qs = self.agent.q_without_communication(self.hidden_states, roles)
        ori_evidence = torch.clamp(ori_qs, 0, torch.inf)
        ori_alpha = ori_evidence + 1
        ori_S = torch.sum(ori_alpha, dim=-1, keepdim=True)
        ori_u = self.args.n_actions / ori_S
        u_err = cmb_u - ori_u

        if u_err.shape[0] == self.args.batch_size:
            true_label = self.generate_true_label(u_err, send_target)
            loss = self.crit_fun(true_label, send_prob)
            self.msg_optim.zero_grad()
            loss.backward()
            self.msg_optim.step()

        return (agents_out, q_without_comm) if counterfactual else agents_out

    def generate_true_label(self, u_err, send_target):
        u_err = u_err.repeat(1, 1, send_target.shape[-1])
        true_label = torch.where(u_err > self.args.min_uncer, 1, 0)
        return true_label.float()

    def init_hidden(self, batch_size):
        self.hidden_states = []
        for agent_id in range(self.n_agents):
            role_idx = self.agent_roles[agent_id]
            role_name = self.args.role_names[role_idx]
            h_dim = self.args.roles[role_name].get("hidden_dim", 64)
            h = torch.zeros(batch_size, h_dim, device=self.args.device)
            self.hidden_states.append(h)
        self.comm_history = None
        self.past_usefulness = None

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.agent_roles = other_mac.agent_roles

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = HeterogeneousAgent(input_shape, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
