import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import random


class collabmodel(torch.nn.Module):
    def __init__(self, movie_size, user_size, embedding_size):
        super(collabmodel, self).__init__()
        self.bias_movie = torch.nn.Parameter(
            torch.zeros(movie_size).unsqueeze(0))
        self.bias_user = torch.nn.Parameter(
            torch.zeros(user_size).unsqueeze(0))
        self.embedding_movie = torch.nn.Embedding(movie_size, embedding_size)
        self.embedding_user = torch.nn.Embedding(user_size, embedding_size)
        self.sig_layer = torch.nn.Sigmoid()

    def forward(self, users, movies):
        movie_total = self.embedding_movie(movies)
        user_total = self.embedding_user(users)
        user_total.unsqueeze_(0).transpose_(0, 1)
        movie_total.unsqueeze_(0).transpose_(0, 1).transpose_(1, 2)
        raw_predict = user_total.matmul(movie_total).squeeze(
        )+self.bias_movie[:, movies].squeeze()+self.bias_user[:, users].squeeze()
        squashed_predict = self.sig_layer(raw_predict)*5.25
        return squashed_predict.squeeze()


class learner():
    def __init__(self, model, loss, users_train, users_test, movies_train, movies_test, ratings_train, ratings_test):
        self.model = model
        self.loss = loss
        self.users_train = users_train
        self.users_test = users_test
        self.movies_train = movies_train
        self.movies_test = movies_test
        self.ratings_train = ratings_train
        self.ratings_test = ratings_test
        self.current_epoch = None
        self.writer = SummaryWriter('runs/{}'.format(random.randint(0, 1e9)))
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9)

    def learn(self, lr=None):
        prediction = self.model.forward(
            self.users_train.to(device), self.movies_train.to(device))
        calculated_loss = self.loss(prediction, self.ratings_train.to(device))
        calculated_loss.backward()
        if self.current_epoch % 100 == 0:
            self.validate()
            self.writer.add_scalar(
                'Training loss', calculated_loss, self.current_epoch)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return calculated_loss

    def validate(self):
        Validation = loss(model(self.users_test.to(device),
                                self.movies_test.to(device)), self.ratings_test.to(device))
        self.writer.add_scalar(
            'Validation loss', Validation, self.current_epoch)
        self.writer.add_histogram("Movie Bias", self.model.bias_movie)
        self.writer.add_histogram(
            "Movie Bias Grad", self.model.bias_movie.grad)
        self.writer.add_histogram("User Bias", self.model.bias_user)
        self.writer.add_histogram("User Bias Grad", self.model.bias_user.grad)
        self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'])
        # print("lr: {}".format(optimizer))    return Validation

    def loop(self, epochs):
        for self.current_epoch in range(epochs):
            loss = self.learn()
        return loss

    def lr_find(self):
        results_array = []
        torch.save(self.model.state_dict(),"./temp.pt")
        for i in range(-6,6):
            # self.current_epoch = 1
            # baseline = self.learn()
            self.model.load_state_dict(torch.load("./temp.pt"))
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=0.1, momentum=0.9)
            torch.save(self.model.state_dict(),"./temp2.pt")
            b = torch.load("./temp2.pt")
            a = torch.load("./temp.pt")
            results_array.append(self.loop(2).cpu())
            # print("was {}, is {}".format(baseline,improved))
            print("hello")
        return results_array

# the nasty glue code section
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# open up the data and grab the top X movies.
header = ['user', 'item', 'rating', 'timestamp']
df = pd.read_csv('./ml-100k/u.data', delimiter='\t', header=None, names=header)
most_common_movies = df.item.value_counts()[:20].index.tolist()
# create a smaller df with the reduced dataset, just assume all users are present
shortdf = df[df['item'].isin(most_common_movies)]
relative_dict = {movie: i for i, movie in enumerate(most_common_movies)}
shortdf['relative_index'] = [relative_dict[movie]
                             for movie in shortdf['item'].values]
#
X_train, X_test, y_train, y_test = train_test_split(
    shortdf[['user', 'relative_index']].values, shortdf['rating'].astype('float').values)
# TODO: use pytorch dataloaders, this is messy
users_train = torch.tensor(X_train[:, 0])
movies_train = torch.tensor(X_train[:, 1])
ratings_train = torch.tensor(y_train).float()

users_test = torch.tensor(X_test[:, 0])
movies_test = torch.tensor(X_test[:, 1])
ratings_test = torch.tensor(y_test).float()

loss = torch.nn.MSELoss()

model = collabmodel(len(most_common_movies), len(df['user'].unique())+1, 30)

model.to(device)
collab_learn = learner(model, loss, users_train, users_test,
                       movies_train, movies_test, ratings_train, ratings_test)

collab_learn.lr_find()
# collab_learn.loop(400)

