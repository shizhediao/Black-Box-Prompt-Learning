conda create -n bdpl python=3.9 -y
conda activate bdpl
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.6.0
pip install -U huggingface_hub
pip install accelerate==0.5.1
pip install datasets==2.11
pip install wandb
pip install scikit-learn
pip install openai