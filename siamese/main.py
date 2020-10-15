import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import time
import numpy as np
import sys
from collections import deque
import os
import argparse

from omniglot_dataset import OmniglotTrain, OmniglotTest
from model import Siamese

def main(args):

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    # train_dataset = dset.ImageFolder(root=Flags.train_path)
    # test_dataset = dset.ImageFolder(root=Flags.test_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # train path: ../omniglot/python/images_background 
    # test path: ../omniglot/python/images_evaluation
    trainSet = OmniglotTrain(args.train_path[0], transform=data_transforms)
    testSet = OmniglotTest(args.test_path[0], transform=transforms.ToTensor(), times = args.times, way = args.way)
    
    # instantiate trainLoader and testLoader
    testLoader = DataLoader(testSet, batch_size=args.way, shuffle=False, num_workers=args.workers)
    trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # this loss combines a Sigmoid layer and the BCELoss in one single class
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # instantiate the Siamese net
    net = Siamese()

    if args.cuda:
        net.cuda()

    net.train()

    learning_rate = args.lr
    optimizer = torch.optim.Adam(net.parameters(),lr = args.lr )
    optimizer.zero_grad()

    train_loss = []
    train_acc = []
    loss_val = 0
    max_length_q = 20
    time_start = time.time()
    queue = deque(maxlen=max_length_q)

    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
        if batch_id > args.max_iter:
            break
        if args.cuda:
            img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
        else:
            img1, img2, label = Variable(img1), Variable(img2), Variable(label)
        optimizer.zero_grad()

        # forwarding img1 and img2
        output = net.forward(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()

        # args.show_every = 10
        if batch_id % args.show_every == 0 :
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s'%(batch_id, loss_val/args.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        
        # args.save_every = 100
        if batch_id % args.save_every == 0:
            torch.save(net.state_dict(), args.model_path[0] + '/model-inter-' + str(batch_id+1) + ".pt")
        
        # args.test_every = 100
        if batch_id % args.test_every == 0:
            right, error = 0, 0
            for _, (test1, test2, label) in enumerate(testLoader, 1):
                if args.cuda:
                    test1, test2, label = test1.cuda(), test2.cuda(), label.cuda()

                # img1 and img2    
                test1, test2, label = Variable(test1), Variable(test2), Variable(label)
                output = net.forward(test1, test2).data.cpu().numpy()
                # output size is 20
                pred = np.argmax(output)
                # pred size is 1

                if pred == 0:
                    right += 1
                else: error += 1

            print('*'*20)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\taccuracy:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
            print('*'*20)
            queue.append(right*1.0/(right+error))
            accuracy = right*1.0/(right+error)
            train_acc.append(accuracy)
            
            learning_rate = learning_rate * 0.99
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        # append loss    
        train_loss.append(loss_val)
        


    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)
    
    with open('train_acc', 'wb') as f:
        pickle.dump(train_acc, f)

    acc = 0.0
    for d in queue:
        acc += d
    print("#"*20)
    print("final accuracy: ", acc/max_length_q)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Code to train Siamese network")

    parser.add_argument("--cuda", help="use", action='store_true')

    parser.add_argument("--train-path", default="./", nargs="+", metavar="TRAIN_PATH", help="Path to the training dataset", type=str)
    parser.add_argument("--test-path", default="./", nargs="+", metavar="TEST_PATH", help="Path to the training dataset", type=str)
    parser.add_argument("--model-path", default="./", nargs="+", metavar="MODEL_PATH", help="Path to the saved models", type=str)

    parser.add_argument("--way", default=20, help="Number of ways", type=int)
    parser.add_argument("--times", default=400, help="Number of samples to test accuracy", type=int)
    parser.add_argument("--workers", default=4, help="Number of dataloader workers", type=int)
    parser.add_argument("--batch-size", default=128, help="Number of batch size", type=int)

    parser.add_argument("--lr", default=0.00006, help="Learning rate", type=float)

    parser.add_argument("--show-every", default=10, help="show result after each show_every iter", type=int)
    parser.add_argument("--test-every", default=100, help="test result after each test_every iter", type=int)
    parser.add_argument("--save-every", default=10000, help="save model after each save_every iter", type=int)

    parser.add_argument("--max-iter", default=80000, help="number of iterations before stopping", type=int)
    parser.add_argument("--gpu-ids", default=0, help="gpu ids used for training", type=str)

    args = parser.parse_args()

    main(args)