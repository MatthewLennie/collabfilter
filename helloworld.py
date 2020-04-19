print("Hello Server World")
import pandas as pd
import torch
import numpy as np
header = ['user','item','rating','timestamp']
df = pd.read_csv('./ml-100k/u.data',delimiter='\t',header=None,names = header)
most_common_movies = df.item.value_counts()[:20].index.tolist()

shortdf = df[df['item'].isin(most_common_movies)]
relative_dict = {movie:i for i,movie in enumerate(most_common_movies)}
shortdf['relative_index']= [relative_dict[movie] for movie in shortdf['item']]
bias_movie = torch.tensor(np.ones(len(most_common_movies)),requires_grad=True)
bias_user =  torch.tensor(np.ones(df['user'].unique().shape[0]+1),requires_grad=True)
embedding_movie = torch.nn.Embedding(len(most_common_movies), 30)
embedding_user = torch.nn.Embedding(df['user'].unique().shape[0]+1, 30)
print(df.count())
weighted_loss = 5
lr = 0.005
for epoch in range(1):
    lr = 0.9*lr
    for row in shortdf.iterrows():
        movie_total = embedding_movie(torch.tensor(row[1]['relative_index']))+bias_movie[row[1]['relative_index']]
        user_total = embedding_user(torch.tensor(row[1]['user']))+bias_user[row[1]['user']]
        raw_predict=movie_total.matmul(user_total)
        
        sig_layer = torch.nn.Sigmoid()
        squashed_predict = sig_layer(raw_predict)*5.25
        # print(squashed_predict)
        loss = torch.nn.MSELoss()
        calculated_loss = loss(squashed_predict,torch.tensor(row[1]['rating'].astype('float')))
        # print(calculated_loss)
        calculated_loss.backward()
        weighted_loss = weighted_loss*0.99 +0.01*calculated_loss.data 
        print(weighted_loss)
        with torch.no_grad():
            embedding_user.weight[row[1]['user']]-= lr*embedding_user.weight.grad[row[1]['user']]
            embedding_movie.weight[row[1]['relative_index']]-= lr*embedding_movie.weight.grad[row[1]['relative_index']]
            bias_movie[row[1]['relative_index']]-= lr*bias_movie.grad[row[1]['relative_index']]
            bias_user[row[1]['user']]-=lr*bias_user.grad[row[1]['user']]
            embedding_movie.weight.grad.zero_()
            embedding_user.weight.grad.zero_()
            bias_user.grad.zero_()
            bias_movie.grad.zero_()
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