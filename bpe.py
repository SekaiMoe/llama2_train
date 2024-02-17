from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# 创建一个空的 BPE 模型
tokenizer = Tokenizer(BPE(unl_token="<unk>"))

# 初始化一个 trainer，设定你想要的参数
trainer = BpeTrainer(special_tokens=["<unk>", "<pad>", "<cls>", "<sep>", "<mask>"])

# 准备一些文本
files = ["path_to_your_text_file_1", "path_to_your_text_file_2", ...]

# 训练 tokenizer
tokenizer.train(files, trainer)

# 保存 tokenizer
tokenizer.save("tokenizer.bin")
