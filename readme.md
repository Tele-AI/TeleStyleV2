# <div align="center">TeleStyle V2: Beyond Content-Preserving Style Transfer with Self-Distillation and Distribution-Matching-Distillation</div>
<div align="center">
    Shiwen Zhang, Yifan Xu, Haibin Huang, Chi Zhang, Xuelong Li
    <br>
    Institute of Artificial Intelligence, China Telecom (TeleAI) 
</div>
<br>
<div align="center">
    [<a href="https://witcherofresearch.github.io/TeleStyleV2/" target="_blank">Project Page</a>]
    [<a href="https://arxiv.org/abs/2606.20709" target="_blank">Paper</a>]
    [<a href="https://huggingface.co/Tele-AI/TeleStyleV2" target="_blank">Model</a>]
    [<a href="https://github.com/Tele-AI/TeleStyleV2" target="_blank">Code</a>]
    [<a href="https://huggingface.co/spaces/witcherderivia/TeleStyleV2" target="_blank">Demo</a>]
</div>

## Abstract
Given a content reference and a style reference, content-preserving style transfer requires the model to generate stylized outputs with content and  style consistency. We introduced TeleStyle V1 to tackle this problem. However, TeleStyle V1 is trained with photorealistic content reference and artistic style reference, which makes it incapable to cope with artistic content reference and realistic style reference in most cases. In this paper, we designed a Self-Distillation data synthesis strategy to construct such triplets from TeleStyle V1.  Trained with such self-distilled triplets, our TeleStyle V2 supports Content-Style references in the forms of Realistic-and-Realistic (RnR), Realistic-and-Stylized (RnS), Stylized-and-Realistic (SnR), Stylized-and-Stylized (SnS). In addition, we found Distribution Matching Distillation could  preserve the general text-guided image editing capability of the foundation model and fix the content consistency degradation caused by SFT process. Through quantitative evaluations, our TeleStyleV2-QIE-2509-DMD performs at least on par with  Qwen-Image-Edit-2509-DMD, demonstrating strong general image editing skills beyond content-preserving style transfer. We observed the content/style reference order confusion problem in TeleStyle V1 and further introduced prompt enhancer to solve it. TeleStyle V2 uses Qwen-Image-Edit's VLM encoder, Qwen2.5-VL-7B, to generate content prompt and style prompt for free. TeleStyle V2 could achieve comparable style transfer performance with state-of-the-art commercial model, gemini-3-pro-image-preview.

 ![alt text](./static/images/compare.png)
## Latest News
- Jan 11, 2026: We release a [free online demo for TeleStyleV2-QIE-2509-Lora-DMD ](https://huggingface.co/spaces/witcherderivia/TeleStyleV2) for diffusers model.  We also release the diffsynth version of [TeleStyleV2-QIE-2509-Lora](https://huggingface.co/Tele-AI/TeleStyleV2/blob/main/diffsynth-TeleStyleV2-QIE-2509-Lora-bf16.safetensors). Please light a star to support this project if you find the demo useful.
- June 10, 2026: We release the <a href="http://arxiv.org/abs/2601.20175" target="_blank">technical report and project page </a> of TeleStyleV2. We release the diffusers version of  [TeleStyleV2-QIE-2509-Lora](https://huggingface.co/Tele-AI/TeleStyleV2/blob/main/diffusers-TeleStyleV2-QIE-2509-Lora-bf16.safetensors). 
- March, 2026: We finished training of TeleStyleV2.
## Schedule


- [ ] Release TeleStyleV2 FLUX 2 models.
- [ ] Release TeleStyleV2-QIE2511-Full models.
- [ ] Release TeleStyleV2-QIE2511-Lora models.
- [ ] Release TeleStyleV2-QIE2509-Full models.
- [x] Release TeleStyleV2-QIE2509-Lora-DMD Demo.
- [x] Release TeleStyleV2-QIE2509-Lora models for diffusers and diffsynth.
- [x] Release Project Page.



## How to use

### 1. Installation

```
pip install -r requirements.txt
```

This environment is tested with:
- Python 3.12
- PyTorch 2.9.1
- diffusers 0.38.0
- transformers 4.57.3
- qwen-vl-utils 0.0.14

on Nvidia H100 80GB GPU. We noted that transformers version has a strong impact towards the generated image. For example, if transformers==5.5.4, the result could be different. We are working on reducing the GPU memory for inference.

### 2. Inference

We provide Gradio inference code. Please note that there are minor changes in the inference logic of QwenImageEditPlusPipeline from the original diffusers implementation.

#### diffusers
```
python gradio_telestylev2_QIE2509_dmd_diffusers.py.
```






## Citation
If you find TeleStyle useful in your research, please light a star for the project and cite our paper, thank you:
```bibtex
@article{telestylev2,
  title={TeleStyle V2: Beyond Content-Preserving Style Transfer with Self-Distillation and Distribution-Matching-Distillation},
  author={Shiwen Zhang, Yifan Xu, Haibin Huang, Chi Zhang, Xuelong Li},
  journal={TeleAI},
  year={2026},
}
```
