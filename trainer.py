from model import RNN
from path_generator import HoleDataset, MazeDataset
from tqdm.autonotebook import tqdm
import torch
import torch.optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

lightblue = '#A6B6FF'


class Trainer:
    def __init__(self, model, optimizer, scheduler, train_dataset, val_dataset, checkpoint_path):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.scheduler = scheduler

    def fit(self, num_epochs, num_train_batch, num_val_batch):
        min_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        for epoch in tqdm(range(1, num_epochs + 1)):
            self.epoch = epoch
            train_loss = self.train(num_train_batch)
            history['train_loss'].append(train_loss)

            display.clear_output(wait=True)
            val_loss = self.validate(num_val_batch)
            history['val_loss'].append(val_loss)

            fn = 'epoch_{}.ckpt'.format(epoch)
            self.save_checkpoint(fn)
            
            if val_loss < min_loss:
                min_loss = val_loss
                fn = 'best.ckpt'
                self.save_checkpoint(fn)

            self.scheduler.step()

            fn = os.path.join(self.checkpoint_path, 'train_loss.npy')
            np.save(fn, np.array(history['train_loss']))
            fn = os.path.join(self.checkpoint_path, 'val_loss.npy')
            np.save(fn, np.array(history['val_loss']))

        return history

    def train(self, num_batch, device='cuda:0'):
        self.model.train()
        with tqdm(range(num_batch)) as t:
            for batch_idx in t:
                t.set_description('BATCH {}'.format(batch_idx))
                velocities, positions, initial_position = self.train_dataset.__next__()
                steps = velocities.shape[1]

                velocities = velocities.to(device)
                positions = positions.to(device)
                initial_position = initial_position.to(device)

                self.optimizer.zero_grad()
                loss = 0

                predict = initial_position
                for step in range(steps):
                    vel = velocities[:, step]
                    pos = positions[:, step]

                    predict = self.model.step(predict, vel)
                    loss += F.mse_loss(predict, pos)

                loss /= steps
                t.set_postfix(loss=loss.item())
                t.update()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    def validate(self, num_batch, device='cuda:0'):
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batch)):
                velocities, positions, initial_position = self.val_dataset.__next__()
                velocities = velocities.to(device)
                positions = positions.to(device)
                initial_position = initial_position.to(device)
                
                predict = initial_position
                prediction = np.zeros((self.val_dataset.batch_size, self.val_dataset.steps, 2))
                for step in range(self.val_dataset.steps):
                    vel = velocities[:, step]
                    pos = positions[:, step]

                    predict = self.model.step(predict, vel)
                    prediction[:, step] = predict.cpu().numpy()
                    loss += F.mse_loss(predict, pos)

                loss /= self.val_dataset.steps
            loss /= num_batch
            tqdm.write('Test Loss: {:7f}'.format(loss.item()))
            self.save_fig(prediction, positions, name='epoch_{}'.format(self.epoch))
            self.create_halfspace_plot(self.model, self.val_dataset.bounds)

        return loss.item()

    def save_checkpoint(self, fn):
        save_path = os.path.join(self.checkpoint_path, fn)
        torch.save(self.model.state_dict(), save_path)

    def save_fig(self, prediction, positions, name=None):
        fig = plt.figure(figsize=(10, 10))

        for i in range(50):
            plt.plot(prediction[i, :, 0], prediction[i, :, 1], color=lightblue)
            plt.plot(positions.cpu()[i, :, 0], positions.cpu()[i, :, 1], color='#AAA')

        if isinstance(self.val_dataset, HoleDataset):
            for bound in self.val_dataset.bounds:
                plt.plot([bound[0], bound[1]], [bound[2], bound[2]], color='k')
                plt.plot([bound[1], bound[1]], [bound[2], bound[3]], color='k')
                plt.plot([bound[0], bound[1]], [bound[3], bound[3]], color='k')
                plt.plot([bound[0], bound[0]], [bound[2], bound[3]], color='k')

        if isinstance(self.val_dataset, MazeDataset):
            for edge in self.val_dataset.edges:
                plt.plot([edge[0], edge[2]], [edge[1], edge[3]], color='k')

        plt.xticks(np.linspace(-1, 1, 21))
        plt.yticks(np.linspace(-1, 1, 21))
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])

        box = self.val_dataset.arena_sz
        plt.plot([box[0], box[1]], [box[2], box[2]], color='k')
        plt.plot([box[1], box[1]], [box[2], box[3]], color='k')
        plt.plot([box[0], box[1]], [box[3], box[3]], color='k')
        plt.plot([box[0], box[0]], [box[2], box[3]], color='k')

        fn = name + '_' if name else ''
        path = os.path.join(self.checkpoint_path, fn + 'paths.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def create_halfspace_plot(self, model, bounds, vel=[-0.01, 0], linewidth=0.5):
        # Create local variables from loaded weights
        W_x = model.w_x.weight.detach().cpu().numpy()
        b_1 = model.w_x.bias.detach().cpu().numpy()
        W_v = model.w_v.weight.detach().cpu().numpy()

        # Initialize the figure
        x = np.linspace(-1, 1, 201)
        y = np.linspace(-1, 1, 201)
        X = np.array(np.meshgrid(x, y)).copy()

        # Generate half-space lines
        x = np.array([-1.1, 1.1])
        y = -W_x[:, 0]/W_x[:, 1]*x[:, None] - b_1/W_x[:, 1] - W_v @ np.array(vel)/W_x[:, 1]

        fig = plt.figure(figsize=(10, 10))
        
        for i in range(y.shape[1]):
            if W_x[i, 1] != 0:
                plt.plot(x, y[:, i], color=plt.cm.Blues(0.2), linewidth=linewidth)
            else:
                plt.plot(np.ones(2) * -b_1[i]/W_x[i, 0], [-1.1, 1.1], color=plt.cm.Blues(0.2), linewidth=linewidth)

        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])

        plt.xticks([])
        plt.yticks([])

        box = [-1, 1, -1, 1]
        plt.plot([box[0], box[1]], [box[2], box[2]], color='k')
        plt.plot([box[1], box[1]], [box[2], box[3]], color='k')
        plt.plot([box[0], box[1]], [box[3], box[3]], color='k')
        plt.plot([box[0], box[0]], [box[2], box[3]], color='k')


        for bound in bounds:
            plt.plot([bound[0], bound[1]], [bound[2], bound[2]], color='k')
            plt.plot([bound[1], bound[1]], [bound[2], bound[3]], color='k')
            plt.plot([bound[0], bound[1]], [bound[3], bound[3]], color='k')
            plt.plot([bound[0], bound[0]], [bound[2], bound[3]], color='k')

        plt.show()
        plt.close()


def main():
    model = RNN()
    model = model.to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 60], gamma=0.1)
    train_dataset = HoleDataset(batch_size=128, dataset_size=128, steps=5)
    val_dataset = HoleDataset(batch_size=512, dataset_size=5120, steps=20)
    
    ckpt_path = os.path.abspath('./checkpoints')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    trainer = Trainer(model, optimizer, scheduler, train_dataset, val_dataset,
                      checkpoint_path='./checkpoints')
    history = trainer.fit(num_epochs=70, num_train_batch=1000, num_val_batch=10)


if __name__ == '__main__':
    main()
