import sentencepiece as spm

# 设置训练文件的路径和输出模型名
input_file = "your_training_data.txt"
model_prefix = "spm_model"
vocab_size = 32000  # 可以根据需要设置词汇表大小

# 训练 SentencePiece 分词器
spm.SentencePieceTrainer.train(input=input_file, model_prefix=model_prefix, vocab_size=vocab_size)

# 这将生成两个文件：'spm_model.model' 和 'spm_model.vocab'
# '.model' 文件是训练好的模型，'.vocab' 文件包含词汇表
