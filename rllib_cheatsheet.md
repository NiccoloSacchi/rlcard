## Start Tensorboard
```console
$ tensorboard --logdir=~/ray_results --host=0.0.0.0
```

## Visualize Ray jobs parallelism
Download the ray timeline file in python with
```python
ray.timeline(filename='mytimeline.json')
```
Then load the created file in:
chrome://tracing/

## Confusional
```python
# Size of batches collected from each worker.
"rollout_fragment_length": 200,
# Number of timesteps collected for each SGD round. This defines the size
# of each SGD epoch.
"train_batch_size": 4000,
# Total SGD batch size across all devices for SGD. This defines the
# minibatch size within each epoch.
"sgd_minibatch_size": 128,
# Number of SGD iterations in each outer loop (i.e., number of epochs to
# execute per train batch).
"num_sgd_iter": 30,
```

## Others
```python
p = trainer.get_policy()

# --- get the policy's model ---
p.model
# <ray.rllib.models.tf.fcnet_v2.FullyConnectedNetwork at 0x7f5aec28cad0>

p.model.base_model.summary()
# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
#  ...

# --- get model's weights ---
trainer.get_policy().get_weights()
# {'default_policy/fc_1/kernel': array([[ ...

# --- get preprocessors ---
trainer.workers.local_worker().preprocessors
# {'default_policy': <ray.rllib.models.preprocessors.TupleFlatteningPreprocessor at 0x7f5aec28c550>}
```
