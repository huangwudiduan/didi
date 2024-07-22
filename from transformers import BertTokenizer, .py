from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_word_embedding(word):
    # 将单词转为token
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取BERT的最后一层隐藏状态
    last_hidden_state = outputs.last_hidden_state
    # 平均池化，得到单词的词向量
    word_embedding = last_hidden_state.mean(dim=1).squeeze()
    return word_embedding.numpy()

def are_same_class(word1, word2, threshold=0.8):
    # 获取两个单词的词向量
    embedding1 = get_word_embedding(word1)
    embedding2 = get_word_embedding(word2)
    # 计算余弦相似度
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity > threshold, similarity

word1 = "apple"
word2 = "orange"
same_class, similarity = are_same_class(word1, word2)
print(f"Word1: {word1}, Word2: {word2}, Same Class: {same_class}, Similarity: {similarity}")