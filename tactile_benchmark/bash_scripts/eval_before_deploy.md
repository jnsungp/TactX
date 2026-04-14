1. change to jit
```
python export_act_to_jit.py \
  --checkpoint_path /home/anyaz/Workspace/tactile/tactile_benchmark/checkpoints/pnp_flexi_test_vision_ckpt/policy_last.ckpt \
  --config_path /home/anyaz/Workspace/tactile/tactile_benchmark/tactile_policy/configs/models/act_flexi_test.yaml \
  --output_path /home/anyaz/Workspace/tactile/tactile_benchmark/checkpoints/pnp_flexi_test_vision_ckpt/policy_jit.pt
```

`python export_act_to_jit.py --policy-name pnp_flexi_test_vision --policy-config act_flexi_test`


2. serve policy
`python serve_act_policy.py --policy_name pnp_eflesh_wide --policy_config act_eflesh_wide`

3. run eval_before_deploy replay eval 
`python eval_before_deploy.py --hdf_file_path /home/anyaz/Workspace/tactile/tactile_benchmark/data/recordings/processed/pnp_flexi_test/episode_0004.hdf5 --plot --tactile_modaity array`