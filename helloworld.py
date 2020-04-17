print("Hello Server World")
import pandas as pd
import torch
header = ['user','item','rating','timestamp']

df = pd.read_csv('./ml-100k/u.data',delimiter='\t',header=None,names = header)
most_common_movies = df.user.value_counts()[:10].index.tolist()
shortdf = df[df['item'].isin(most_common_movies)]
embedding_movie = torch.nn.Embedding(len(most_common_movies), 3)
embedding_user = torch.nn.Embedding(shortdf['user'].unique().shape[0], 3)
print(df.count())
raw_predict= embedding_movie(torch.tensor([0])).matmul(embedding_user(torch.tensor([0])).transpose(0,1))
sig_layer = torch.nn.Sigmoid()
squashed_predict = sig_layer(raw_predict)*5.25
print(squashed_predict)
loss = torch.nn.MSELoss()
calculated_loss = loss(squashed_predict,torch.tensor([4.5]))
print(calculated_loss)
calculated_loss.backward()
print(embedding_user.weight[0].grad)
embedding_user
# x =torch.tensor([[1.],[2.]],requires_grad=True)
# y = torch.tensor([[1.,1.],[2.,2.],[3.,3.],[4.,4.]],requires_grad=True)
# # print(y*x)
# # print(x.size())
# for i in range(1):
#     z =(y.matmul(x).sum()-10)**2
#     z.backward()
#     print(y.grad)
#     print(x.grad)
#     print(z)
#     with torch.no_grad():
#         x -= 0.00001*x.grad.data
#         # x.grad.zero_()
#     print(x)

#     print(z)