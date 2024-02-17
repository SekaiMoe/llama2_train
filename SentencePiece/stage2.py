import os
import sentencepiece as spm

# 创建一个SentencePiece模型实例
sp = spm.SentencePieceProcessor(model_file='spm_model.model')

# 编码为token的ID
tokens_ids = sp.encode('这是一个例句', out_type=int)

# 解码为原始文本
original_text = sp.decode(tokens_ids)
