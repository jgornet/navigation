from model import RNN
from path_generator import HoleDataset
from tqdm.autonotebook import tqdm
import torch
import torch.optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

darkblue = (0, 0.08, 0.45)
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
        for epoch in tqdm(range(1, num_epochs + 1)):
            self.epoch = epoch
            self.train(num_train_batch)
            val_loss = self.validate(num_val_batch)

            fn = 'epoch_{}.ckpt'.format(epoch)
            self.save_checkpoint(fn)
            
            if val_loss < min_loss:
                min_loss = val_loss
                fn = 'best_{:4f}.ckpt'.format(val_loss)
                self.save_checkpoint(fn)

            self.scheduler.step()

    def train(self, num_batch, device='cuda:0'):
        self.model.train()
        with tqdm(range(num_batch)) as t:
            for batch_idx in t:
                t.set_description('BATCH {}'.format(batch_idx))
                velocities, positions, initial_position = self.train_dataset.__next__()

                velocities = velocities.to(device)
                positions = positions.to(device)
                initial_position = initial_position.to(device)

                self.optimizer.zero_grad()
                loss = 0

                predict = initial_position
                for step in range(self.train_dataset.steps):
                    vel = velocities[:, step]
                    pos = positions[:, step]

                    predict = self.model.step(predict, vel)
                    loss += F.mse_loss(predict, pos)

                loss /= self.train_dataset.steps
                t.set_postfix(loss=loss.item())
                t.update()
                loss.backward()
                self.optimizer.step()

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

        plt.xticks(np.linspace(-1, 1, 21))
        plt.yticks(np.linspace(-1, 1, 21))
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])

        fn = name + '_' if name else ''
        path = os.path.join(self.checkpoint_path, fn + 'paths.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    model = RNN()
    model = model.to('cuda:0')
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 35], gamma=0.1)
    train_dataset = HoleDataset(batch_size=128, steps=5)
    val_dataset = HoleDataset(batch_size=512, steps=20)
    
    ckpt_path = os.path.abspath('./checkpoints')
    trainer = Trainer(model, optimizer, scheduler, train_dataset, val_dataset,
                      checkpoint_path='./checkpoints')
    trainer.fit(num_epochs=40, num_train_batch=1000, num_val_batch=10)


if __name__ == '__main__':
    main()
