import torch
import numpy as np
import pandas as pd

class collabmodel(torch.nn.Module):
    def __init__(self,movie_size,user_size,embedding_size):
        super(collabmodel, self).__init__()
        self.bias_movie = torch.tensor([np.ones(movie_size)],requires_grad=True,device=device)
        self.bias_user =  torch.tensor([np.ones(user_size)],requires_grad=True,device=device)
        self.embedding_movie = torch.nn.Embedding(movie_size,embedding_size)
        self.embedding_user = torch.nn.Embedding(user_size,embedding_size)
        self.sig_layer = torch.nn.Sigmoid()

    def forward(self, users,movies):
        movie_total = self.embedding_movie(movies)
        user_total = self.embedding_user(users)
        user_total.unsqueeze_(0).transpose_(0,1)
        movie_total.unsqueeze_(0).transpose_(0,1).transpose_(1,2)
        raw_predict=user_total.matmul(movie_total).squeeze()+self.bias_movie[:,movies].squeeze()+self.bias_user[:,users].squeeze()
        squashed_predict = self.sig_layer(raw_predict)*5.25
        return squashed_predict.squeeze()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#open up the data and grab the top X movies. 
header = ['user','item','rating','timestamp']
df = pd.read_csv('./ml-100k/u.data',delimiter='\t',header=None,names = header)
most_common_movies = df.item.value_counts()[:200].index.tolist()
# create a smaller df with the reduced dataset, just assume all users are present
shortdf = df[df['item'].isin(most_common_movies)]
relative_dict = {movie:i for i,movie in enumerate(most_common_movies)}
shortdf['relative_index']= [relative_dict[movie] for movie in shortdf['item'].values]
# 
users = torch.tensor(shortdf['user'].values)
movies = torch.tensor(shortdf['relative_index'].values)
ratings = torch.tensor(shortdf['rating'].astype('float').values)
loss = torch.nn.MSELoss()
# calculated_loss = loss(squashed_predict,torch.tensor(row[1]['rating'].astype('float')))
# print(calculated_loss)
model = collabmodel(len(most_common_movies),len(df['user'].unique())+1,30)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

for i in range(600000):
    calculated_loss = loss(model.forward(users.to(device),movies.to(device)),ratings.to(device))
    calculated_loss.backward()
    print(calculated_loss)
    optimizer.step()
    optimizer.zero_grad()

