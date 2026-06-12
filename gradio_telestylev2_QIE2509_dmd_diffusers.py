import gradio as gr
import numpy as np
import random
import torch
import spaces

from PIL import Image
#from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwen_vl_utils import process_vision_info



import os

from huggingface_hub import  hf_hub_download

def update_textbox(selected_items):
    # Join the selected list of strings into a comma-separated string
    return ", ".join(selected_items)


pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipe.to('cuda')
pipe.set_progress_bar_config(disable=None)


'''
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
        download_source='huggingface',
        origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
        download_source='huggingface',origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
        download_source='huggingface',origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
    download_source='huggingface',origin_file_pattern="processor/"),
)
'''





qwenstyle= hf_hub_download(repo_id="Tele-AI/TeleStyleV2", filename="diffusers-TeleStyleV2-QIE-2509-Lora-bf16.safetensors")
speedup = hf_hub_download(repo_id="Tele-AI/TeleStyleV2", filename="QIE-2509-Lightning-4steps-V1.0-bf16.safetensors")



pipe.load_lora_weights(
    qwenstyle,adapter_name='style'
)


pipe.load_lora_weights(
    speedup,adapter_name='dmd'
)

pipe.set_adapters(["style", "dmd",], adapter_weights=[1.0, 1.0])
pipe.fuse_lora(adapter_names=["style", "dmd"], lora_scale=1.0)
pipe.unload_lora_weights()






dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"




MAX_SEED = np.iinfo(np.int32).max


@spaces.GPU(size="xlarge")


def infer(
    content_ref,
    style_ref,
    prompt,
    seed=123,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    minedge=1024,
    progress=gr.Progress(track_tqdm=True),
    checkbox=[],
    
):
    
    

    

    
    content_text_input='describe main objects (fewer than 3) with separated words, each word is separated by comma,  the total number of words is strictly fewer than 3'
    style_text_input='describe only the artistic style, material and stroke, lighting, color in 5 words, not objects.'
    #pipe.text_encoder.eval()
    content_prompt=''
    style_prompt=''

    
    

    

    if content_ref is not None:
        content_ref=Image.fromarray(content_ref)
        content_messages = [
        {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": content_ref,
                    },
                    {"type": "text", "text": content_text_input},
                ],
            }
        ]
        content_text = pipe.processor.apply_chat_template(
            content_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(content_messages)
        inputs = pipe.processor(
            text=[content_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        # Inference: Generation of the output
        generated_ids = pipe.text_encoder.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        content_prompt = pipe.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f"content_prompt={content_prompt}")
    if style_ref is not None:
        style_ref=Image.fromarray(style_ref)
        style_messages = [
        {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": style_ref,
                    },
                    {"type": "text", "text": style_text_input},
                ],
            }
        ]
        style_text = pipe.processor.apply_chat_template(
            style_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(style_messages)
        inputs = pipe.processor(
            text=[style_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        # Inference: Generation of the output
        generated_ids = pipe.text_encoder.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        style_prompt = pipe.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f"style_prompt={style_prompt}")
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    
    
    
    
    sw,sh,w,h=0,0,0,0
    if content_ref:
        w,h=content_ref.size



        #minedge=1024
        if w>h:
            r=w/h
            h=minedge
            w=int(h*r)-int(h*r)%16
            
        else:
            r=h/w
            w=minedge
            h=int(w*r)-int(w*r)%16
    if style_ref:
        sw,sh=style_ref.size
        if sw>sh:
            r=sw/sh
            sh=minedge
            sw=int(sh*r)-int(sh*r)%16
            
        else:
            r=sh/sw
            sw=minedge
            sh=int(sw*r)-int(sw*r)%16


    
    
    
    print(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale},")
    
    if content_ref and style_ref:
        images = [
            content_ref.resize((w, h)),
            style_ref.resize((sw, sh)) ,
            #style_ref.resize((minedge, minedge)) ,
        ]
    elif content_ref:
        images = [
            content_ref.resize((w, h)),
            #style_ref.resize((sw, sh)) ,
            #style_ref.resize((minedge, minedge)) ,
        ]
    elif style_ref:
        images = [
            #content_ref.resize((w, h)),
            style_ref.resize((sw, sh)) ,
            #style_ref.resize((minedge, minedge)) ,
        ]
    
    if "infer with content prompt" in checkbox and content_prompt not in prompt:
        prompt=','.join([prompt,content_prompt])
    if "infer with style prompt" in checkbox and style_prompt not in prompt:
        prompt=','.join([prompt,style_prompt])
    if "infer with content prompt" not in checkbox and content_prompt in prompt:
        prompt=prompt.replace(content_prompt.strip(','),'')
    if "infer with style prompt" not in checkbox and style_prompt in prompt:
        prompt=prompt.replace(style_prompt.strip(),'')
    prompt=prompt.strip(',')
    print(f"Calling pipeline with prompt: '{prompt}'")
    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": true_guidance_scale,
        "negative_prompt": " ",
        "num_inference_steps": num_inference_steps,
        "guidance_scale": true_guidance_scale,
        "num_images_per_prompt": 1,
        "width": w or sw,
        "height": h or sh, 
    }
    with torch.inference_mode():
        image = pipe(**inputs)
    image = image.images[0]
    
    

    
    




    return image, seed, content_prompt, style_prompt, prompt

# --- Examples and UI Layout ---
examples = []



_HEADER_ = '''
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">TeleStyle V2</h1>
    
</div>


<p style="font-size: 1rem; margin-bottom: 1.5rem;">Paper: <a href='https://witcherofresearch.github.io/TeleStyleV2' target='_blank'>TeleStyle V2: Beyond Content-Preserving Style Transfer with Self-Distillation and Distribution-Matching-Distillation</a> | Codes: <a href='https://github.com/Tele-AI/TeleStyleV2' target='_blank'>GitHub</a></p>
<p style="font-size: 1rem; margin-bottom: 1.5rem;">Update: prompt enhancer provided, and the model supports content ref/style ref only input, which means you could use the model as an image editing model and style transfer model at the same time. So you don't have to provide a style reference now, the model also accepts prompt for style transfer, which makes the model more flexible. If you choose infer with content/style prompt, do not forget to clean the prompt box when you run new  inference.</p>

<p style="font-size: 1rem; margin-bottom: 1.5rem;">If you encounter an Error with this demo, the most possible reason is ZeroGPU out-of-memory and the solution is to decrease the Min Edge of the generated image from 1024 to a lower value.  </p>
'''  

with gr.Blocks() as demo:

    with gr.Column(elem_id="col-container"):
        
        gr.Markdown(_HEADER_)
        gr.Markdown("This is a demo of TeleStyle V2.")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    content_ref = gr.Image(label="content ref", type="numpy", )
                    style_ref = gr.Image(label="style ref", type="numpy", )
                    #print(f"type(content_ref)={type(content_ref)}")
                    
                #input_images = gr.Gallery(label="Input Images", show_label=False, type="pil", interactive=True)
            

            result = gr.Image(label="Result", show_label=True, type="pil")
            #result = gr.Gallery(label="Result", show_label=True, type="pil")
        with gr.Column():
            
            checkbox=gr.CheckboxGroup(["infer with content prompt", "infer with style prompt"], label="Prompt Enhancer", )
            content_prompt=gr.Text(
                    label="Content Reference Prompt",
                    show_label=True,
                    container=True,
            )
            style_prompt=gr.Text(
                    label="Style Reference Prompt",
                    show_label=True,
                    container=True,
            )
            prompt = gr.Text(
                    label="Prompt",
                    value='Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.',
                    show_label=True,
                    placeholder='Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.',
                    container=True,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=True):
            # Negative prompt UI element is removed here

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=123,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=False)

            with gr.Row():

                true_guidance_scale = gr.Slider(
                    label="CFG should be 1.0",
                    minimum=0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps should be 4",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=4,
                )
                
                minedge = gr.Slider(
                    label="Min Edge of the generated image",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=1024,
                )
        with gr.Row(), gr.Column():
            gr.Markdown("## Examples")
            gr.Markdown("changing the minedge could lead to different style similarity.")
            default_prompt='Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.'
            gr.Examples(examples=[
                ['./qwenstyleref/content_1.webp','./qwenstyleref/style_1.jpg','',123,False,1.0,4,1024,[]],
                ['./qwenstyleref/content_6.jpg','./qwenstyleref/style_6.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/style_6.png','./qwenstyleref/content_6.jpg','',123,False,1.0,4,1024,["infer with style prompt"]],
                ['./qwenstyleref/content_3.png','./qwenstyleref/style_3.png','',123,False,1.0,4,1024,[]],
                ['./qwenstyleref/content_4.png','./qwenstyleref/content_7.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/content_7.png','./qwenstyleref/content_4.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/content_9.jpg','./qwenstyleref/style_9.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/style_9.png','./qwenstyleref/content_9.jpg',default_prompt,123,False,1.0,4,1024,["infer with style prompt"]],
                ['./qwenstyleref/content_11.png','./qwenstyleref/style_11.jpg',default_prompt,123,False,1.0,4,832,[]],
                ['./qwenstyleref/content_9.jpg',None,"convert to photorealistic photograph",123,False,1.0,4,1024,[]],
                ],
                inputs=[content_ref,
                    style_ref,
                    prompt,
                    seed,
                    randomize_seed,
                    true_guidance_scale,
                    num_inference_steps,
                    minedge,
                    checkbox
                    ], 
                outputs=[result, seed, content_prompt, style_prompt,prompt], 
                fn=infer, 
                cache_examples=False
                )        
                
                
                
                

        # gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False)

    gr.on(
        triggers=[run_button.click],
        fn=infer,
        inputs=[
            content_ref,
            style_ref,
            prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
            minedge,
            checkbox,
            
        ],
        outputs=[result, seed, content_prompt, style_prompt,prompt],
    )

    
    

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')
'''
['./qwenstyleref/pulpfiction_2.jpg','./qwenstyleref/styleref=6_style_ref.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/styleref=0_content_ref.png','./qwenstyleref/110.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/romanholiday_1.jpg','./qwenstyleref/s0099____1113_01_query_1_img_000146_1682705733350_08158389675901344.jpg.jpg',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/styleref=0_content_ref.png','./qwenstyleref/125.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/fallenangle.jpg','./qwenstyleref/styleref=s0038.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/styleref=0_content_ref.png','./qwenstyleref/styleref=s0572.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/startrooper1.jpg','./qwenstyleref/david-face-760x985.jpg','Style Transfer Figure  1 into marble material.',123,False,1.0,4,1024,[]],
                ['./qwenstyleref/startrooper1.jpg','./qwenstyleref/125.png',default_prompt, 123,False,1.0,4,1024,[]],
                ['./qwenstyleref/possession.png','./qwenstyleref/s0026____0907_01_query_0_img_000194_1682674358294_041656249089406583.jpeg.jpg',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/styleref=0_content_ref.png','./qwenstyleref/Jotarokujo.webp',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/wallstreet1.jpg','./qwenstyleref/034.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/bird.jpeg','./qwenstyleref/styleref=s0539.png',default_prompt,123,False,1.0,4,1024,[]],
'''
