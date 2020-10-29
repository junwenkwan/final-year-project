import  torch
import os
import  numpy as np
from    omniglotNShot import OmniglotNShot
import  argparse
from    meta import Meta
import pickle

def create_config_file():
    config = [
        ('conv2d', [64, 1, 3, 3, 1, 0]),
        ('max_pool2d', [2, 2, 0]),
        ('bn', [64]),
        ('relu', [True]),

        ('conv2d', [64, 64, 3, 3, 1, 0]),
        ('max_pool2d', [2, 2, 0]),
        ('bn', [64]),
        ('relu', [True]),

        ('conv2d', [64, 64, 3, 3, 1, 0]),
        ('max_pool2d', [2, 2, 0]),
        ('bn', [64]),
        ('relu', [True]),
        
        ('conv2d', [64, 64, 1, 1, 1, 0]),
        ('bn', [64]),
        ('relu', [True]),
        
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]    

    return config

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = create_config_file()

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    omniglot_data_train = OmniglotNShot(args.root_dataset, batchsz=args.task_num, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, imgsz=args.imgsz)

    train_acc = []
    train_loss = []

    for iteration in range(args.epoch):

        x_support, y_support, x_query, y_query = omniglot_data_train.next()
        x_support, y_support, x_query, y_query = torch.from_numpy(x_support).to(device), torch.from_numpy(y_support).to(device), \
                                     torch.from_numpy(x_query).to(device), torch.from_numpy(y_query).to(device)

        # x_support.shape: torch.Size([16, 5, 1, 28, 28])
        # y_support.shape: torch.Size([16, 5])
        # x_query.shape: torch.Size([16, 75, 1, 28, 28])
        # y_query.shape: torch.Size([16, 75])

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs, loss_q = maml(x_support, y_support, x_query, y_query)

        if iteration % args.display_every == 0:
            print('Epochs:', iteration, '\ttraining acc:', accs)

        if iteration % args.test_every == 0:
            accs = []
            for _ in range(1000//args.task_num):
                # test
                x_support, y_support, x_query, y_query = omniglot_data_train.next('test')
                x_support, y_support, x_query, y_query = torch.from_numpy(x_support).to(device), torch.from_numpy(y_support).to(device), \
                                             torch.from_numpy(x_query).to(device), torch.from_numpy(y_query).to(device)

                # split to single task each time
                for x_support_one, y_support_one, x_query_one, y_query_one in zip(x_support, y_support, x_query, y_query):
                    test_acc = maml.finetunning(x_support_one, y_support_one, x_query_one, y_query_one)
                    accs.append(test_acc)

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test Acc:', accs)

        train_acc.append(accs[-1])
        train_loss.append(loss_q)

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)
    
    with open('train_acc', 'wb') as f:
        pickle.dump(train_acc, f)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=50000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=16)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--root_dataset', type=str, help='root to omniglot dataset', default='/home/jun/fypv2/omniglot/python')
    argparser.add_argument('--test_every', type=int, help='testing', default=1000)
    argparser.add_argument('--display_every', type=int, help='dispaly', default=50)

    args = argparser.parse_args()

    main(args)
