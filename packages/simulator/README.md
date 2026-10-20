# simulator (physicalai)

Physical AI-specific simulator config layer built on top of the upstream `leisaac` package.

Contents:

- `src/simulator/tasks/template/single_arm_franka_cfg.py` — shared single-arm Franka base config
  (scene, cameras, observations, `use_joint_position_actions()`, LeRobot frame builder)
- `src/simulator/tasks/cup_stacking/` — `HCIS-CupStacking-SingleArm-v0`
- `src/simulator/tasks/cutlery_arrangement/` — `HCIS-CutleryArrangement-SingleArm-v0`
- `src/simulator/tasks/toy_blocks_collection/` — `HCIS-ToyBlocksCollection-SingleArm-v0`
- `src/simulator/tasks/external.py` — resolves `--task` given as a gym id, a `.py` file, or `module:Class`
- `src/simulator/utils/domain_randomization.py` — reset events for object poses and lighting
- `src/simulator/utils/object_pose_recording.py` — LeRobot recorder that also writes `object_pose.<name>` columns
- `assets/` — scene and object USDs
