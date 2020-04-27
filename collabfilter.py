import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import random
import collections

class collabmodel(torch.nn.Module):
    """[collaborative filtering model, very basic, handles mmult some what manually]
    """
    def __init__(self, movie_size, user_size, embedding_size):
        """[summary]

        Arguments:
            torch {[torch.nn.Module]} -- [inheritance]
            movie_size {[int]} -- [number of movies]
            user_size {[int]} -- [number of users]
            embedding_size {[int]} -- [depth of embedding (3 to 4 seems good for this example)]
        """
        super(collabmodel, self).__init__()
        self.bias_movie = torch.nn.Parameter(
            torch.zeros(movie_size).unsqueeze(0))
        self.bias_user = torch.nn.Parameter(
            torch.zeros(user_size).unsqueeze(0))
        self.embedding_movie = torch.nn.Embedding(movie_size, embedding_size)
        self.embedding_user = torch.nn.Embedding(user_size, embedding_size)
        self.sig_layer = torch.nn.Sigmoid()

    def forward(self, users, movies):
        """[forward prop step]

        Arguments:
            users {[int]} -- [user indicies]
            movies {[int]} -- [movie indecies]

        Returns:
            [torch.tensor] -- [the predicitns from the forward]
        """
        movie_total = self.embedding_movie(movies)
        user_total = self.embedding_user(users)
        user_total.unsqueeze_(0).transpose_(0, 1)
        movie_total.unsqueeze_(0).transpose_(0, 1).transpose_(1, 2)
        raw_predict = user_total.matmul(movie_total).squeeze(
        )+self.bias_movie[:, movies].squeeze()+self.bias_user[:, users].squeeze()
        squashed_predict = self.sig_layer(raw_predict)*5.25
        return squashed_predict.squeeze()


class learner():
    """[handes the boiler plate code for training the model, i.e. learning loops]
    """
    def __init__(self, model, loss, users_train, users_test, movies_train, movies_test, ratings_train, ratings_test):
        """[initializes variables and the tensorboard summary writer]

        Arguments:
            model {[collabmodel]} -- [collabmodel object]
            loss {[torch loss]} -- [i.e. mse]
            users_train {[torch.tensor]} -- [training dataset of users]
            users_test {[torch.tensor]} -- [test dataset of users]
            movies_train {[torch.tensor]} -- [training dataset of movies]
            movies_test {[torch.tensor]} -- [test dataset of movies]
            ratings_train {[torch.tensor]} -- [training dataset of ratings]
            ratings_test {[torch.tensor]} -- [test dataset of ratings]
        """
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
        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=0.05, momentum=0.9)
        self.init_optimizer()
    def learn(self):
        """[calcuates loss, backpropagates, takes step]

        Returns:
            [calculated_loss] -- [the loss before the optimizer takes a step]
        """
        prediction = self.model.forward(
            self.users_train.to(device), self.movies_train.to(device))
        calculated_loss = self.loss(prediction, self.ratings_train.to(device))
        calculated_loss.backward()
        if self.current_epoch % 2 == 0:
            self.validate()
            self.writer.add_scalar(
                'Training loss', calculated_loss, self.current_epoch)
        self.schedule.step()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return calculated_loss

    def init_optimizer(self):
        """[helper function to initialize optimizer, using this to do lr annealing restarts]

        Returns:
            [None] -- [None]
        """
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.05)
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max = 200)
        return None

    def validate(self):
        """[gets loss from validation step and adds items to the tensorboard log]

        Returns:
            [torch.tensor] -- [returns errors for each example with their labels for analysis]
        """
        Validation = loss(model(self.users_test.to(device),
                                self.movies_test.to(device)), self.ratings_test.to(device))
        self.writer.add_scalar(
            'Validation loss', Validation, self.current_epoch)
        self.writer.add_histogram("Movie Bias", self.model.bias_movie)
        self.writer.add_histogram(
            "Movie Bias Grad", self.model.bias_movie.grad)
        self.writer.add_histogram("User Bias", self.model.bias_user)
        self.writer.add_histogram("User Bias Grad", self.model.bias_user.grad)
        self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'],self.current_epoch)
        errors = model(self.users_test.to(device),
                       self.movies_test.to(device)) - self.ratings_test.to(device)
        self.writer.add_histogram("validation errors", errors)
        return errors, self.movies_test

    def loop(self, epochs):
        """[helper function to repeat learning process]

        Arguments:
            epochs {[int]} -- [number of learning steps, batch mode steps = epochs]

        Returns:
            [float] -- [loss and end of training]
        """
        for self.current_epoch in range(epochs):
            loss = self.learn()
            if self.schedule._step_count %200 ==0:
                self.init_optimizer()
        return loss

    def lr_find(self):
        """[performs a lr find by trying different learning steps, similar to FASTAI's LRFinder, crude implementation use carefully]

        Returns:
            [torch.tensor] -- [returns the loss after a number of steps with each lr]
        """
        results_array = []
        lr_array = []
        torch.save(self.model.state_dict(), "./temp.pt")
        for i in range(-6, 6):
            self.model.load_state_dict(torch.load("./temp.pt"))
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=10**i, momentum=0.9)
            results_array.append(self.loop(100).cpu())
            lr_array.append(i)
            print("Result {} @ lr = 10**{}".format(results_array[-1].data, i))

        self.model.load_state_dict(torch.load("./temp.pt"))
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=10**i, momentum=0.9)
        return lr_array, results_array

    def plot_lr(self):
        """[does lr_find, plots the learning rate finder results.]

        Returns:
            [torch.tensor] -- [losses]
        """
        import matplotlib.pyplot as plt
        lr_array, results_array = self.lr_find()
        plt.plot(lr_array, results_array)
        plt.savefig("lr_finder.png")
        return results_array


# the nasty glue code section
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# open up the data and grab the top X movies.
header = ['user', 'item', 'rating', 'timestamp']
df = pd.read_csv('./ml-100k/u.data', delimiter='\t', header=None, names=header)
movies = pd.read_csv('./ml-100k/u.item', delimiter='\|', header=None)
movies.rename(columns={0: 'item'}, inplace=True)
movies[1].astype('str')
# find the most common movies
most_common_movies = df.item.value_counts()[:20].index.tolist()
# create a smaller df with the reduced dataset, just assume all users are present
# relative index just relates the reduced movie index to the original dataset 'item'
shortdf = df[df['item'].isin(most_common_movies)]
shortmovie = movies[movies['item'].isin(most_common_movies)].reset_index()
shortmovie.drop('index', axis=1, inplace=True)
shortmovie.reset_index(inplace=True)
shortmovie.rename(columns={'index': 'relative_index'}, inplace=True)
shortdf = shortdf.merge(shortmovie, on='item', how='inner', validate='m:1')
# test train split
X_train, X_test, y_train, y_test = train_test_split(
    shortdf[['user', 'relative_index']].values, shortdf['rating'].astype('float').values)
# TODO: use pytorch dataloaders, this is messy
# note the dimensionality. Each example is split acorss the batch dimension. 
users_train = torch.tensor(X_train[:, 0])
movies_train = torch.tensor(X_train[:, 1])
ratings_train = torch.tensor(y_train).float()

users_test = torch.tensor(X_test[:, 0])
movies_test = torch.tensor(X_test[:, 1])
ratings_test = torch.tensor(y_test).float()

loss = torch.nn.MSELoss()
#create the model
model = collabmodel(len(most_common_movies), len(df['user'].unique())+1, 4)
model.to(device)
# create the learner
collab_learn = learner(model, loss, users_train, users_test,
                       movies_train, movies_test, ratings_train, ratings_test)

collab_learn.loop(50)
# show best and worst movie biases
shortmovie['bias'] = collab_learn.model.bias_movie.squeeze(
).cpu().detach().numpy()
collab_learn.writer.add_text('best_movies', shortmovie.sort_values(
    'bias').tail(10)[['bias', 1]].to_markdown())
collab_learn.writer.add_text('worst_movies', shortmovie.sort_values(
    'bias').head(10)[['bias', 1]].to_markdown())

# get the lossses of the validation set
losses, labels = collab_learn.validate()
# create a dataframe with the losses per example
losses_df = pd.DataFrame(
    [losses.to('cpu').detach().numpy(), shortmovie.loc[labels][1]]).T
losses_df['abs_loss'] = abs(losses_df[0])
losses_df = losses_df.infer_objects()
#group into movies and push the summary statistics to the tensorboard. 
collab_learn.writer.add_text('movie losses descriptions', losses_df.groupby(
    1).describe()[0].sort_values('std').to_markdown())

# can be useful for analysis but leaving out for now. 
# for i in range(20):
#     collab_learn.writer.add_text(
#         'worst underpredictions', '{}'.format(losses_df.sort_values(0).iloc[i]))
#     collab_learn.writer.add_text(
#         'worst overpredictions', '{}'.format(losses_df.sort_values(0).iloc[-i]))
#     collab_learn.writer.add_text('best predictions', '{}'.format(
#         losses_df.sort_values('abs_loss').iloc[i]))
# a = 1
