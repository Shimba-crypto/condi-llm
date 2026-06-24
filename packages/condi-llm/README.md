`condi-llm`
``Distributed training and serving framework for models up to 405B parameters.``

`Install`
```pip install condi-llm```
``Quick start``
```python

import condi_llm as cllm

model = cllm.AutoModel.from_pretrained("condi-70b")
trainer = cllm.Trainer(model, strategy="fsdp_shard", grad_accum=32)
trainer.fit(dataset, epochs=3)
```

``License
Apache-2.0``
