mkdir -p data

s="http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz"
d="data/inception_resnet_v2_2016_08_30.ckpt"

if [ -f "$d" ]; then
    echo "[!] Model has already been downloaded."
    exit
fi

# Download pre-trained model from Tensorflow.
wget "$s" -O "$d.tar.gz"

# Extract model
cd $(dirname "$d")
tar xzvf "$(basename "$d").tar.gz"
rm "$(basename "$d").tar.gz"
