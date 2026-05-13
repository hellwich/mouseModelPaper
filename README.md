# Mouse Simulation

mouse_sim: Simulates 3D mice skeletons moving in a cage depending on current behavior. Estimates behavior from 3D skeleton movement. Renders multi-view video of moving mice.

free3D...: The mouse mesh model that has been used in this work can be found here: https://free3d.com/3d-model/mouse-rigged-5650.html

dlc_mouse: Tracks 2D body parts in video.

mouse_seg: Tracks mouse semantic segment in video.

dlc_mouse_cleaning: Uses 3D reasoning (including 3D model) to clean the 2D body part observations and does simple single point (upto) 3-view bundle adjustment based using Ceres.

mouseba: Unfinished. Untested. Ceres bundle adjustment with model constraint and 3D path smoothness constraint over time.

LockBoxModel: lock box model for 3D printing

In every directory there is a protocol.txt file with the most important commands executed. Most recent ones can be found at the end of the files.

## Sequence of Commands

List of major program calls for the basic processing workflow:

To try run the commands please look up the required files in the repository and fix the paths to the files as required on your own system.

### Simulation of skeleton movements
  python mouse_sim2.py --mouse-graph mouse3_mesh_shortTail.txt --write-video --duration 180 --fps 10 --seed 45 --out-dir out2m_5 --export-cameras

### Rendering of videos, generation of DLC training data
  python mouse_deform_render_multi_swaprb_dlc_checkvid_visfix_with_seg_color_and_labelid.py  --coords-3d out2m_5/coords_3d.csv   --cameras out2m_5/cameras.json   --mesh ../free3DMouserigged3Dmodel/mouse.obj   --mesh-nodes mouse_mesh_nodes.txt   --out-dir dlc_synth_project_seg5 --mouse-on-top   --dlc-project   --dlc-every 5   --dlc-scorer synthetic --dlc-overwrite --dlc-check-video --dlc-occ-front-tol 10 --dlc-vis-patch 2 --seg-video --seg-id-video --seg-id-ext mkv --seg-id-fourcc FFV1   --seg-id-vis-video --N Polar_Bears_2004-11-15.jpg --S A_view_on_Sams_sand_dunes.jpeg --W Beach_at_Fort_Lauderdale.jpg --E Mt._Everest_from_Gokyo_Ri_November_5,_2012.jpg --T Sky_with_puffy_clouds.JPG --B 1000_F_1851230750_Hs5HaJxALwRZZnT6q2Gdjx286LAWVsm0.jpg

### DLC training
  python train_dlc_multi_mouse_engine.py   --config /media/hellwich/Herakles/hellwich/hellwich/sdk3/mouseModel/mouse_sim/dlc_synth_project/config.yaml

### DLC prediction
  python predict_dlc_multi_mouse_batch_v3.py   --config /media/hellwich/Herakles/hellwich/hellwich/sdk3/mouseModel/mouse_sim/dlc_synth_project/config.yaml  --videos-dir /media/hellwich/Herakles/hellwich/hellwich/sdk3/mouseModel/mouse_sim/out2m_2_render  --destfolder /media/hellwich/Herakles/hellwich/hellwich/sdk3/mouseModel/mouse_sim/out2m_2_render/dlc_predictions  --n-tracks 2   --batchsize 8   --create-labeled-video

### Semantic Segmentation (extraction, training and prediction)
conda env is mouse_seg_clean

  python mice_seg_v3.py extract --rgb_video /media/hellwich/Herakles/hellwich/hellwich/sdk3/mouseModel/mouse_sim/dlc_synth_project_seg4/videos/cam1_top.mp4   --mask_video /media/hellwich/Herakles/hellwich/hellwich/sdk3/mouseModel/mouse_sim/dlc_synth_project_seg4/videos/cam1_top_seg_id.mkv   --out_rgb_dir /media/hellwich/Herakles/hellwich/hellwich/sdk3/mouseModel/mouse_sim/dlc_synth_project_seg4/videos/frames/vid001_top_rgb   --out_mask_dir /media/hellwich/Herakles/hellwich/hellwich/sdk3/mouseModel/mouse_sim/dlc_synth_project_seg4/videos/frames/vid001_top_mask   --every 1
  
  python mice_seg_v3.py train   --manifest ./manifest2.csv   --out_dir ./runs/run2   --oversample_signals interaction,mouse0_rear,mouse1_rear   --oversample_factor 4.0   --camera_crop   --camera_crop_margin 10 --resize_mode letterbox --val_split 0.3333333 --seed 7
  
  python mice_seg_v3_fullres.py  predict --ckpt ./runs/run2/ckpt_best.pt   --rgb_source /home/hellwich/hellwich/sdk3/mouseModel/mouse_sim/out2m_2_render/cam1_top.mp4   --view top   --out_dir ./preds/out2m_2/top   --cameras_json /home/hellwich/hellwich/sdk3/mouseModel/mouse_sim/out2m_2/cameras.json   --camera_crop --camera_crop_margin 10   --resize_mode letterbox --save_fullres_masks --out_mask_video ./preds/out2m_2/cam1_top.mp4

### DLC cleanup using 3D modeling
  python run_mouse3d.py   --cameras ../../mouse_sim/out2m_2/cameras.json   --dlc ../../mouse_sim/out2m_2_render/dlc_long.csv   --seg-dir ../../mouse_seg/preds/out2m_2   --out ../dlc_cleaned/out2m_2/tuneB --graph_model ../../mouse_sim/mouse3_mesh_shortTail.txt --graph-rigid-tol-mm-max 5 --debug

### Behavior estimation training
  python train_behavior_pair_xlstm_enabled.py   --coords-3d out_pair/coords_3d.csv   --model lstm   --out-dir out_pair_train2 --amp --tf32
  python train_behavior_pair_xlstm_enabled.py   --coords-3d out_pair/coords_3d.csv   --model transformer   --out-dir out_pair_train2 --amp --tf32
  python train_behavior_pair_xlstm_enabled.py   --coords-3d out_pair/coords_3d.csv   --model baseline_rule   --out-dir out_pair_train2 --amp --tf32
  python train_behavior_pair_xlstm_enabled.py   --coords-3d out_pair/coords_3d.csv   --model baseline_logreg   --out-dir out_pair_train2 --amp --tf32
  python train_behavior_pair_xlstm_enabled.py   --coords-3d out_pair/coords_3d.csv   --model xlstm   --out-dir out_pair_train2 --amp --tf3

### Behavior prediction
  (mouse_sim_fix)$ python predict_behavior_pair_xlstm.py   --ckpt out_pair_train2/behavior_pair_lstm.pt   --coords-3d converted_coords_3d.csv   --out-dir out2m_2_pred/from_imputed/lstm   --report

## Installation

In every subdirectory there is an environment.yml to create a conda environment.

