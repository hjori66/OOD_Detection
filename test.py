import torch 
import numpy as np

from model import ResNet34
from utils import *
from tqdm import tqdm


def ood_test_baseline(model, id_train_loader, id_test_loader, ood_test_loader, args):
    """
    Implementation of baseline ood detection method
    """
    threshold = 0.67
    
    model = model.cuda()
    model.eval()

    TPR = 0.
    TNR = 0.
    ID_ACC = 0.
    OOD_ACC = 0.

    with torch.no_grad():
        for x, y in id_test_loader:
            x, y = x.cuda(), y.cuda()
            
            pred, feature_list = model(x)
            confidence_score, pred_class = torch.max(torch.softmax(pred, dim=1), dim=1)
            TPR += (confidence_score > threshold).sum().item() / id_test_loader.batch_size

            ID_ACC += (((pred_class - y == 0).int()).sum()).item() / id_test_loader.batch_size
        
        for x, y in ood_test_loader:
            x, y = x.cuda(), y.cuda()
            
            pred, feature_list = model(x)
            confidence_score, pred_class = torch.max(torch.softmax(pred, dim=1), dim=1)
            TNR += (confidence_score < threshold).sum().item() / ood_test_loader.batch_size

            OOD_ACC += (((pred_class - y == 0).int()).sum()).item() / ood_test_loader.batch_size
        
    print('TPR: {:.4}% |TNP: {:.4}% |threshold: {}'.format(TPR / len(id_test_loader) * 100, TNR / len(ood_test_loader) * 100, threshold))
    print('ID_ACC:{:.4}% |OOD_ACC: {:.4}%'.format(ID_ACC / len(id_test_loader), OOD_ACC / len(ood_test_loader)))
            

def mahalanobis_distance(pred, class_mu, inv_tied_cov):
    scores = []
    for mu in class_mu:
        # delta = pred - torch.repeat_interleave(mu.view(1, -1), repeats=pred.size(0), dim=0)
        delta = pred - mu[None:, ]
        # check1 = torch.einsum('bc,cx->bx', delta, inv_tied_cov)
        # check2 = torch.einsum('bx,bx->b', check1, delta)
        # scores.append(-check2)
        scores.append(
            -torch.diag(torch.matmul(torch.matmul(delta, inv_tied_cov), delta.t()), 0))
    return torch.stack(scores)


def ood_test_mahalanobis(model, id_train_loader, id_test_loader, ood_test_loader, args):
    """
    - step 1. calculate empircal mean and covariance of each of class conditional Gaussian distibtuion(CIFAR10 has 10 classes)
        - If you don't use feature ensemble, performance will be degenerated, but whether to use it is up to you.
        - If you don't use input pre-processing, performance will be degenerated, but whether to use it is up to you.
    """
    model = model.cuda()
    model.train()

    feature_num = -2
    num_class = 10
    num_dim = 10
    num_data_per_class = 5000

    with torch.no_grad():
        total_feature = []
        for _ in range(num_class):
            total_feature.append(list())

        for x, y in id_train_loader:
            x, y = x.cuda(), y.cuda()

            """ Why Wrong..?
            pred, _ = model(x)
            for i, label in enumerate(y.cpu().detach().numpy()):
                total_pred[label].append(pred.cpu().detach().numpy()[i])
            """

            for i in range(num_class):
                pred, feature_list = model(x[y == i])
                if feature_num == 0:
                    total_feature[i].extend(pred)
                else:

                    total_feature[i].extend(feature_list[feature_num])

        """ Why Wrong..?
        class_mu = []
        for i, class_pred in enumerate(total_pred):
            class_mu.append((np.stack(class_pred)).mean(axis=0))
        class_mu = np.stack(class_mu)
        """

        class_mu = []
        feature_vectors = []
        for i, class_pred in enumerate(total_feature):
            feature_vector = torch.cat(class_pred, dim=0).view(num_data_per_class, num_class)
            feature_vectors.append(feature_vector)
            class_mu.append(feature_vector.mean(dim=0))

        class_cov = []
        for i, class_pred in enumerate(total_feature):
            feature_vector = feature_vectors[i]
            mu = class_mu[i]

            new_feature_vector = feature_vector - mu[None, :]
            cov = torch.zeros(num_class, num_class, dtype=torch.float).cuda()
            for row in new_feature_vector:
                row = row.unsqueeze(1)
                cov += torch.matmul(row, row.t())
            class_cov.append(cov / num_data_per_class)
        class_covs = torch.cat(class_cov).view(num_class, num_dim, num_dim)

        tied_cov = class_covs.mean(axis=0)
        # inv_tied_cov = np.linalg.inv(tied_cov)
        inv_tied_cov = tied_cov.inverse()

    """
    - step 2. calculate test samples' confidence score
                by using Mahalanobis distance and just calculated parameters of class conditional Gaussian distributions
    - step 3. calculate empircal mean and covariance of each of class conditional Gaussian distibtuion(CIFAR10 has 10 classes) 
    """
    model = model.cuda()
    model.eval()

    TPR = 0.
    TNR = 0.
    ID_ACC = 0.
    OOD_ACC = 0.

    for x, y in id_test_loader:
        x, y = x.cuda(), y.cuda()

        if args.noisy_preprocessing:
            x.requires_grad = True
            pred, feature_list = model(x)
            md = mahalanobis_distance(pred, class_mu, inv_tied_cov).sum()
            md.backward()
            x = x - args.epsilon_noise * x.grad.sign()

        with torch.no_grad():
            pred, feature_list = model(x)

            confidence_score_list = []

            for mean_i in class_mu:
                feature_vector = pred - mean_i[None:, ]
                confidence_score_list.append(-torch.diag(torch.matmul(torch.matmul(feature_vector, tied_cov.inverse()), feature_vector.t()), 0))

            confidence_score_list = torch.stack(confidence_score_list)
            confidence_score, pred_class = torch.max(confidence_score_list, dim=0)
            TPR += (confidence_score > args.threshold).sum().item() / id_test_loader.batch_size

            ID_ACC += (((pred_class - y == 0).int()).sum()).item() / id_test_loader.batch_size

    for x, y in ood_test_loader:
        x, y = x.cuda(), y.cuda()

        if args.noisy_preprocessing:
            x.requires_grad = True
            pred, feature_list = model(x)
            md = mahalanobis_distance(pred, class_mu, inv_tied_cov).sum()
            md.backward()
            x = x - args.epsilon_noise * x.grad.sign()

        with torch.no_grad():
            pred, feature_list = model(x)

            confidence_score_list = []

            for mean_i in class_mu:
                feature_vector = pred - mean_i[None:, ]
                confidence_score_list.append(-torch.diag(torch.matmul(torch.matmul(feature_vector, tied_cov.inverse()), feature_vector.t()), 0))

            confidence_score_list = torch.stack(confidence_score_list)
            confidence_score, pred_class = torch.max(confidence_score_list, dim=0)
            TNR += (confidence_score < args.threshold).sum().item() / ood_test_loader.batch_size

            OOD_ACC += (((pred_class - y == 0).int()).sum()).item() / ood_test_loader.batch_size

    print('TPR: {:.4}% |TNP: {:.4}% |threshold: {}'.format(TPR / len(id_test_loader) * 100,
                                                           TNR / len(ood_test_loader) * 100, args.threshold))
    print('ID_ACC:{:.4}% |OOD_ACC: {:.4}%'.format(ID_ACC / len(id_test_loader), OOD_ACC / len(ood_test_loader)))


def id_classification_test(model, id_train_loader, id_test_loader, args):
    """
    TODO : Calculate test accuracy of CIFAR-10 test set by using Mahalanobis classification method 
    """
    pass


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    def parse_args():
        import argparse
        parser = argparse.ArgumentParser('Mahalanobis-args')
        
        # experimental settings
        parser.add_argument('--seed', type=int, default=0, help='Random seed.')   
        parser.add_argument('--task', type=str, default='ood_detection', help='classification | ood_detection')
        parser.add_argument('--alg', type=str, default='mahalanobis', help='baseline | mahalanobis')

        parser.add_argument('--train_bs', type=int, default=1000, help='Batch size of in_trainloader.')
        parser.add_argument('--test_bs', type=int, default=250, help='Batch size of in_testloader and out_testloader.')

        parser.add_argument('--noisy_preprocessing', type=bool, default=False, help='Noise Pre-processing.')
        parser.add_argument('--epsilon_noise', type=int, default=1e-2, help='Parameter for Noise Pre-processing.')
        parser.add_argument('--threshold', type=int, default=-8, help='Threshold.')
        parser.add_argument('--num_workers', type=int, default=0)

        args = parser.parse_args()

        return args

    # arg parse ood_test_baseline
    args = parse_args()

    # set seed
    set_seed(args.seed)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    """
    in-distribution data loader(CIFAR-10) 
    """
    
    # id_trainloader will be used for estimating empirical class mean and covariance
    id_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    id_trainloader = torch.utils.data.DataLoader(id_trainset, batch_size=args.train_bs,
                                            shuffle=False, num_workers=args.num_workers)

    # id_testloader will be used for test the given ood detection algorithm
    id_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    id_testloader = torch.utils.data.DataLoader(id_testset, batch_size=args.test_bs,
                                            shuffle=False, num_workers=args.num_workers)
    
    """
    out-of-distribtuion data looader(SVHN)
    """

    # ood_testloader will be used for test the given ood detection algorithm
    ood_testset = torchvision.datasets.SVHN(root='./data', split='test',
                                        download=True, transform=transform)
    ood_testloader = torch.utils.data.DataLoader(ood_testset, batch_size=args.test_bs,
                                            shuffle=False, num_workers=args.num_workers)
    
    # load model trained on CIFAR-10 
    model = ResNet34()
    model.load_state_dict(torch.load('./model/resnet34-31.pth'))

    # ood dectection test
    if args.task == 'ood_detection':
        if args.alg == 'baseline':
            print('result of baseline alg')
            ood_test_baseline(model, id_trainloader, id_testloader, ood_testloader, args)
        elif args.alg == 'mahalanobis':
            print('result of mahalanobis alg')
            ood_test_mahalanobis(model, id_trainloader, id_testloader, ood_testloader, args)
        else:
            print('--alg should be baseline or mahalanobis')
    
    # classification test
    elif args.task == 'classification':
        id_classification_test(model, id_trainloader, id_testloader, args)
    else:
        print('--task should be ood_detection or classification')
