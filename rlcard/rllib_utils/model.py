import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from gym.spaces import Box


def flatten_list(list_of_lists):
    flattened = []
    for l in list_of_lists:
        if isinstance(l, list):
            flattened += flatten_list(l)
        else:
            flattened.append(l)
    return flattened


class ParametricActionsModel(TFModelV2):
    """ Parametric model that handles varying action spaces"""
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 true_obs_shape=(24,),
                 action_embed_size=None):
        super(ParametricActionsModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        if action_embed_size is None:
            action_embed_size = action_space.n  # this works for Discrete() action

        self.action_embed_model = FullyConnectedNetwork(
            obs_space=Box(-1, 1, shape=true_obs_shape),
            action_space=action_space,
            num_outputs=action_embed_size,
            model_config=model_config,
            name=name + "_action_embed"
        )
        self.base_model = self.action_embed_model.base_model
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        # Compute the predicted action probabilties
        # input_dict["obs"]["real_obs"] is a list of 1d tensors if the observation space is a Tuple while
        # it should be a tensor. When it is a list we concatenate the various 1d tensors
        obs_concat = input_dict["obs"]["real_obs"]
        if isinstance(obs_concat, list):
            obs_concat = tf.concat(values=flatten_list(obs_concat), axis=1)
        action_embed, _ = self.action_embed_model({"obs": obs_concat})

        # Mask out invalid actions (use tf.float32.min for stability)
        action_mask = input_dict["obs"]["action_mask"]
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_embed + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()
