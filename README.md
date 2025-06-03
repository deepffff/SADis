<!--
 * @Description: Free-Lunch Color-Texture Disentanglement for Stylized Image Generation (SADis)
 * @Date: 2025-03-21 13:34:33
 * @LastEditTime: 2025-06-03 18:28:23
 * @FilePath: \SADis\README.md
-->

# SADis
The implement of the paper **[Free-Lunch Color-Texture Disentanglement for Stylized Image Generation (SADis)](https://arxiv.org/pdf/2503.14275)**.

<div align="center">
    <a href="https://arxiv.org/abs/2503.14275" target="_blank" style="color: pink;">[Arxiv]  </a>&nbsp;&nbsp
    <a href="https://deepffff.github.io/sadis.github.io/" target="_blank" style="color: pink;">[Project page]</a>
</div>

---

## Mehthod
![Framework](/docs/framework.jpg)


## Results
![Visualization Results](/docs/vis_results7-1.jpg)

---

## Try SADis

### **1. Download Pretrained Weights**
```bash
# git clone this repository
git clone https://github.com/deepffff/SADis.git
cd SADis

# download ip-adapter weights into ./models from: https://huggingface.co/h94/IP-Adapter/tree/main/models

# download weights of sdxl into ./sdxl_models from https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models
```

Ensure the directory structure includes the following paths:
- 'models/image_encoder'
- 'sdxl_models/ip-adapter-plus_sdxl_vit-h.bin'



### **2. Set Up the Environment**
```bash
# create new anaconda env
conda env create -f environment.yml
conda activate color_texture
```


### **3. Run Inference**
```bash
python infer_style_plus_color_texture.py
# Note: Adjust hyperparameters as recommended in the comments to achieve better performance.
```




---


## Citation
If you find the project useful, please cite the papers:



```
@misc{qin2025freelunchcolortexturedisentanglementstylized,
  title={Free-Lunch Color-Texture Disentanglement for Stylized Image Generation}, 
  author={Jiang Qin and Senmao Li and Alexandra Gomez-Villa and Shiqi Yang and Yaxing Wang and Kai Wang and Joost van de Weijer},
  year={2025},
  eprint={2503.14275},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.14275}, 
}
```
