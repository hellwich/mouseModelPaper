conda env create -f environment_dlc_tf_gpu.yml
conda activate dlc_mouse

# install hooks
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cp hooks/activate.d/dlc_mouse_env.sh "$CONDA_PREFIX/etc/conda/activate.d/"
cp hooks/deactivate.d/dlc_mouse_env.sh "$CONDA_PREFIX/etc/conda/deactivate.d/"

# re-activate to apply hooks
conda deactivate
conda activate dlc_mouse

# final compatibility override
./post_setup_dlc_tf_gpu.sh dlc_mouse

