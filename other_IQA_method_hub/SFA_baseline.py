
import os

from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from tqdm import tqdm
import numpy as np
import torch.autograd
import config_hub
import creator


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_data, self.test_data = creator.datasets_creator(self.config)
        self.model = creator.model_creator(self.config)
        self.model = self.model.to(device)
        self.model_name = type(self.model).__name__ + self.config.train_description
        self.ckpt_path = config.ckpt_path
        self.start_epoch = 0
        self.max_epoch = self.config.max_epoch

    def fit(self):
        self._train_one_epoch()

    def _train_one_epoch(self):
        # start training
        self.model.eval()
        X1_train, X2_train, X3_train = [], [], []
        y_train = []
        X1_test, X2_test, X3_test = [], [], []
        y_test = []
        sum_index = np.array([1, 2, 3, 4, 5])
        for i in range(len(self.train_data)):
            # print('Extracting features of the {}-th image'.format(i))
            with torch.no_grad():
                features = self.model(self.train_data[i].to(device))
                y_train.append(np.sum(np.multiply(self.train_data.label[i], sum_index)))
                X1_train.append(torch.cat((torch.mean(features, dim=0),
                                           torch.std(features, dim=0))).to('cpu').numpy())
                X2_train.append(torch.cat((torch.mean(features, dim=0),
                                           torch.sqrt(torch.mean(torch.pow(features, 2), dim=0)),
                                           torch.pow(torch.mean(torch.pow(features, 3), dim=0), 1 / 3),
                                           torch.pow(torch.mean(torch.pow(features, 4), dim=0), 1 / 4))).to(
                    'cpu').numpy())
                features = features.to('cpu').numpy()
                X3_train.append(np.concatenate((np.min(features, axis=0),
                                                np.quantile(features, 0.25, axis=0),
                                                np.median(features, axis=0),
                                                np.quantile(features, 0.75, axis=0),
                                                np.max(features, axis=0))))


        for i in range(len(self.test_data)):
            # print('Extracting features of the {}-th image'.format(i))
            with torch.no_grad():
                features = self.model(self.test_data[i].to(device))
                y_test.append(np.sum(np.multiply(self.test_data.label[i], sum_index)))
                X1_test.append(torch.cat((torch.mean(features, dim=0),
                                           torch.std(features, dim=0))).to('cpu').numpy())
                X2_test.append(torch.cat((torch.mean(features, dim=0),
                                           torch.sqrt(torch.mean(torch.pow(features, 2), dim=0)),
                                           torch.pow(torch.mean(torch.pow(features, 3), dim=0), 1 / 3),
                                           torch.pow(torch.mean(torch.pow(features, 4), dim=0), 1 / 4))).to(
                    'cpu').numpy())
                features = features.to('cpu').numpy()
                X3_test.append(np.concatenate((np.min(features, axis=0),
                                                np.quantile(features, 0.25, axis=0),
                                                np.median(features, axis=0),
                                                np.quantile(features, 0.75, axis=0),
                                                np.max(features, axis=0))))


        # # SVR is slow.
        # regr1 = LinearSVR(random_state=0)
        # regr1.fit(X1_train, y_train)
        # y1_pred = regr1.predict(X1_test)
        # regr2 = LinearSVR(random_state=0)
        # regr2.fit(X2_train, y_train)
        # y2_pred = regr2.predict(X2_test)
        # regr3 = LinearSVR(random_state=0)
        # regr3.fit(X3_train, y_train)
        # y3_pred = regr3.predict(X3_test)
        pls10_1 = PLSRegression(n_components=10)
        pls10_1.fit(X1_train, y_train)
        y1_pred = pls10_1.predict(X1_test)
        pls10_2 = PLSRegression(n_components=10)
        pls10_2.fit(X2_train, y_train)
        y2_pred = pls10_2.predict(X2_test)
        pls10_3 = PLSRegression(n_components=10)
        pls10_3.fit(X3_train, y_train)
        y3_pred = pls10_3.predict(X3_test)


        y_pred = (y1_pred + y2_pred + y3_pred) / 3

        y_pred = np.reshape(np.asarray(y_pred), (-1,))
        y_test = np.reshape(np.asarray(y_test), (-1,))
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
        print('SROCC: {} KROCC: {} PLCC: {} RMSE: {}'.format(SROCC, KROCC, PLCC, RMSE))



def main(cfg):
    t = Trainer(cfg)
    t.fit()


if __name__ == "__main__":
    config = config_hub.SFA_config()
    for i in tqdm(range(0, 10)):
        config = config_hub.SFA_config()
        split = i + 1
        config.split = split
        config.ckpt_path = os.path.join(config.ckpt_path, str(config.split))
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        print(config.network, config.train_description)
        main(config)
