import gradio as gr
from fastai.vision.all import *
import pathlib
import re

# Fix for PosixPath issue on Windows
pathlib.PosixPath = pathlib.WindowsPath
def label_func(fname): 
    return re.match(r'(.+)_\d+.jpg$', fname.name).groups()[0]
learn = load_learner('model(finetuned resnet50).pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Pet Breed Classifier"
description = "A pet breed classifier trained on the Oxford Pets dataset with fastai. Created as a demo for Gradio and HuggingFace Spaces."
article="<p style='text-align: center'><a href='https://github.com/akhilpranjal/HostingOnHFSpaces' target='_blank'>Github</a></p>"
examples = ['exampleImages/chihuahua.jpeg','exampleImages/pomeranian.jpg','exampleImages/british-shorthair.webp','exampleImages/bengal.png']

gr.Interface(
	fn=predict,
	inputs=gr.Image(scale=512),
	outputs=gr.Label(num_top_classes=7),
	title=title,
	description=description,
	article=article,
	examples=examples,
).launch(share=True)