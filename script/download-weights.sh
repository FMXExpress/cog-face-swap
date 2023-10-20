# Create necessary directories
mkdir checkpoints
mkdir -p ~/.insightface/models/buffalo_l/

# Download and extract the buffalo_l model
wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
# gdown -O buffalo_l.zip https://drive.google.com/uc?id=1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB
unzip buffalo_l.zip -d ~/.insightface/models/buffalo_l/
rm buffalo_l.zip

# Download GFPGANv1.4
wget -P checkpoints https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
# Download inswapper_128.onnx
wget -P checkpoints  https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx
