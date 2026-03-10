conda create -n dime python=3.10 -y
conda activate dime
conda install conda-forge::faiss
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install --upgrade ir_datasets
pip install pandas
pip install -U sentence-transformers
pip install nvitop
pip install safetensors
pip install python-dotenv
pip install ir_measures

conda activate dime
mkdir -p $CONDA_PREFIX/etc/conda/activate.d;
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh