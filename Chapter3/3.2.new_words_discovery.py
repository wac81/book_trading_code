from smoothnlp.algorithm.phrase import extract_phrase
f=open('./data/新闻集样本10k.txt','r')
lines = f.readlines()
top200=extract_phrase(lines,top_k=200)
print(top200)