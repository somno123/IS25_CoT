# Reasoning-Based Approach with Chain-of-Thought for Alzheimer’s Detection Using Speech and Large Language Models

## Dependencies and Setup
This project is implemented and tested with **Python 3.8**. 

```
conda create --name cot python=3.8
conda activate cot
pip install -r requirements.txt
```

## Pretrained models (ResNetSE34)

```
wget -P ./voxceleb_trainer http://www.robots.ox.ac.uk/~joon/data/baseline_v2_smproto.model
```

## Execution Command

### Basic Usage
```
python main.py
```

## Citation
```
@article{park2025reasoning,
  title={Reasoning-Based Approach with Chain-of-Thought for Alzheimer's Detection Using Speech and Large Language Models},
  author={Chanwoo Park, Anna Seo Gyeong Choi, Sunghye Cho and Chanwoo Kim},
  journal={arXiv preprint arXiv:2506.01683},
  year={2025}
}
```