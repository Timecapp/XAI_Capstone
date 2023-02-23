# -*- coding: utf-8 -*-
"""connect.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zWl4gBX-jTM4d1tRSoyUfG98UHbjvU3d
"""

import transformers

# Load your model
model = transformers.AutoModel.from_pretrained("LuckyApollo/WiscBreastXAI")

# Connect your model to the Hugging Face app
app.connect(model)