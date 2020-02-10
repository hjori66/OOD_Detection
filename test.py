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
    # pred_y = pred.cpu().detach().numpy()
    # delta = np.repeat(pred_y[:, :, np.newaxis], class_mu.shape[0], axis=-1) - np.repeat(class_mu[np.newaxis, :, :], pred.shape[0], axis=0)   # B, C, D
    # delta = np.transpose(delta, [0, 2, 1])   # B, D, C
    # check1 = np.einsum('bdc,cy->bdy', delta, inv_tied_cov)
    # check2 = np.einsum('bdy,bdc->bd', np.einsum('bdc,cy->bdy', delta, inv_tied_cov), delta)

    # class_mu_tensor = torch.cuda.FloatTensor(class_mu)
    # inv_tied_cov_tensor = torch.cuda.FloatTensor(inv_tied_cov)
    # a = torch.repeat_interleave(pred.view(1, -1, 10), repeats=10, dim=0)  # C, B, D
    # a = a.permute(1, 0, 2)
    # b = torch.repeat_interleave(class_mu_tensor.view(1, 10, 10), repeats=pred.size()[0], dim=0)   # B, D, C
    # delta_tensor = a-b
    # check1 = torch.einsum('bdc,xd->bcx', delta_tensor, inv_tied_cov_tensor)
    # check2 = torch.einsum('bcx,bdc->bd', check1, delta_tensor)

    scores = []
    inv_tied_cov_tensor = torch.cuda.FloatTensor(inv_tied_cov)
    for mu in class_mu:
        mu_tensor = torch.cuda.FloatTensor(mu)
        delta_tensor = pred - torch.repeat_interleave(mu_tensor.view(1, -1), repeats=pred.size(0), dim=0)
        check1 = torch.einsum('bc,cx->bx', delta_tensor, inv_tied_cov_tensor)
        check2 = torch.einsum('bx,bx->b', check1, delta_tensor)
        scores.append(-check2)
    return torch.stack(scores, dim=-1)


def ood_test_mahalanobis(model, id_train_loader, id_test_loader, ood_test_loader, args):
    """
    - step 1. calculate empircal mean and covariance of each of class conditional Gaussian distibtuion(CIFAR10 has 10 classes)
        - If you don't use feature ensemble, performance will be degenerated, but whether to use it is up to you.
        - If you don't use input pre-processing, performance will be degenerated, but whether to use it is up to you.
    """
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        total_pred = []
        for _ in range(10):
            total_pred.append(list())

        for x, y in id_train_loader:
            x, y = x.cuda(), y.cuda()

            pred, feature_list = model(x)
            for i, label in enumerate(y.cpu().detach().numpy()):
                total_pred[label].append(pred.cpu().detach().numpy()[i])

    class_mu = []
    for i, class_pred in enumerate(total_pred):
        class_mu.append(np.stack(class_pred).mean(axis=0))
    class_mu = np.stack(class_mu)

    class_cov = []
    for i, class_pred in enumerate(total_pred):
        mu = class_mu[i]
        cov_list = []
        for pred in class_pred:
            y = (pred - mu).reshape(1, -1)
            cov_list.append(np.matmul(y.transpose(), y))
        class_cov.append(np.stack(cov_list).mean(axis=0))
    class_cov = np.array(class_cov)

    tied_cov = class_cov.mean(axis=0)
    inv_tied_cov = np.linalg.inv(tied_cov)

    """
    - step 2. calculate test samples' confidence score
                by using Mahalanobis distance and just calculated parameters of class conditional Gaussian distributions
    - step 3. calculate empircal mean and covariance of each of class conditional Gaussian distibtuion(CIFAR10 has 10 classes) 
    """
    TPR = 0.
    TNR = 0.
    ID_ACC = 0.
    OOD_ACC = 0.
    noisy_preprocessing = True

    for x, y in id_test_loader:
        x, y = x.cuda(), y.cuda()

        if noisy_preprocessing:
            x.requires_grad = True
            pred, feature_list = model(x)
            md = mahalanobis_distance(pred, class_mu, inv_tied_cov).sum()
            md.backward()
            x = x - args.epsilon_noise * x.grad.sign()

        with torch.no_grad():
            pred, feature_list = model(x)

            # confidence_score, pred_class = torch.max(torch.softmax(pred, dim=1), dim=1)
            md = mahalanobis_distance(pred, class_mu, inv_tied_cov)
            confidence_score, pred_class = torch.max(md, axis=1)
            TPR += (confidence_score > args.threshold).sum().item() / id_test_loader.batch_size

            ID_ACC += (((pred_class - y == 0).int()).sum()).item() / id_test_loader.batch_size

    for x, y in ood_test_loader:
        x, y = x.cuda(), y.cuda()

        if noisy_preprocessing:
            x.requires_grad = True
            pred, feature_list = model(x)
            md = mahalanobis_distance(pred, class_mu, inv_tied_cov).sum()
            md.backward()
            x = x - args.epsilon_noise * x.grad.sign()

        with torch.no_grad():
            pred, feature_list = model(x)
            # confidence_score, pred_class = torch.max(torch.softmax(pred, dim=1), dim=1)
            md = mahalanobis_distance(pred, class_mu, inv_tied_cov)
            confidence_score, pred_class = torch.max(md, axis=1)
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
        parser.add_argument('--test_bs', type=int, default=256, help='Batch size of in_testloader and out_testloader.')

        parser.add_argument('--noisy_preprocessing', type=bool, default=True, help='Noise Pre-processing.')
        parser.add_argument('--epsilon_noise', type=int, default=1e-2, help='Parameter for Noise Pre-processing.')
        parser.add_argument('--threshold', type=int, default=-23, help='Threshold.')
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
