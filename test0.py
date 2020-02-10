import torch 
import numpy as np

from model import ResNet34
from utils import * 


def ood_test_baseline(model, id_train_loader, id_test_loader, ood_test_loader, args):
    threshold = 0.67
    
    model = model.cuda()
    model.eval()

    TPR = 0.
    TNR = 0.
    with torch.no_grad():
        for x, y in id_test_loader:
            x, y = x.cuda(), y.cuda()
            
            pred, feature_list = model(x)
            confidence_score, pred_class = torch.max(torch.softmax(pred, dim=1), dim=1)
            TPR += (confidence_score > threshold).sum().item() / id_test_loader.batch_size
        
        for x, y in ood_test_loader:
            x, y = x.cuda(), y.cuda()
            
            pred, feature_list = model(x)
            confidence_score, pred_class = torch.max(torch.softmax(pred, dim=1), dim=1)
            TNR += (confidence_score < threshold).sum().item() / ood_test_loader.batch_size
        
    print('TPR: {:.4}% |TNP: {:.4}% |threshold: {}'.format(TPR / len(id_test_loader) * 100, TNR / len(ood_test_loader) * 100, threshold))
            

def ood_test_mahalanobis(model, id_train_loader, id_test_loader, ood_test_loader, args):
    """
    TODO
    - step 1. calculate empircal mean and covariance of each of class conditional Gaussian distibtuion(CIFAR10 has 10 classes) 
        - If you don't use feature ensemble, performance will be degenerated, but whether to use it is up to you.
        - If you don't use input pre-processing, performance will be degenerated, but whether to use it is up to you.
    - stpe 2. calculate test samples' confidence score by using Mahalanobis distance and just calculated parameters of class conditional Gaussian distributions
    - step 3. compare the confidence score and the threshold. if confidence score > threshold, it will be assigned to in-distribtuion sample.
    """
    # mean_list, cov_list 구하기 
    threshold = args.threshold

    mean_list, cov = get_trained_features(model, id_train_loader, args)
  
    tied_cov = cov / 50000

    # mahalanobis distance를 사용해서 test sample의 class, confidence 계산 
    model = model.cuda()
    model.eval()


    TPR = 0.
    TNR = 0.
    ID_ACC = 0.
    OOD_ACC = 0.
    with torch.no_grad():
       
        for x, y in id_test_loader:
            x = x.cuda()
            y = y.cuda()

            pred, feature_list = model(x)
            
            confidence_score_list = []

            for mean_i in mean_list:
                feature_vector = pred - mean_i[None:, ]
                confidence_score_list.append(torch.diag(torch.matmul(torch.matmul(feature_vector, tied_cov.inverse()), feature_vector.t()), 0))

            confidence_score_list = torch.stack(confidence_score_list)
            confidence_score, pred_class = torch.min(confidence_score_list, dim=0)

            TPR += (confidence_score > threshold).sum().item() / id_test_loader.batch_size
            ID_ACC += (((pred_class - y == 0).int()).sum()).item() / id_test_loader.batch_size
        
        for x, y in ood_test_loader:
            x = x.cuda()
            y = y.cuda()

            pred, feature_list = model(x)
            
            confidence_score_list = []

            for mean_i in mean_list:
                feature_vector = pred - mean_i[None:, ]
                confidence_score_list.append(torch.diag(torch.matmul(torch.matmul(feature_vector, tied_cov.inverse()), feature_vector.t()), 0))

                # delta_tensor = pred - torch.repeat_interleave(mean_i[None:, ], repeats=pred.size(0), dim=0)
                # # delta_tensor = pred - mu[None:, ]
                # check1 = torch.einsum('bc,cx->bx', delta_tensor, tied_cov.inverse())
                # check2 = torch.einsum('bx,bx->b', check1, delta_tensor)
                # confidence_score_list.append(-check2)

            confidence_score_list = torch.stack(confidence_score_list)
            confidence_score, pred_class = torch.min(confidence_score_list, dim=0)

            TNR += (confidence_score < threshold).sum().item() / ood_test_loader.batch_size
            OOD_ACC += (((pred_class - y == 0).int()).sum()).item() / ood_test_loader.batch_size
    
    print('TPR: {:.4}% |TNP: {:.4}% |threshold: {}'.format(TPR / len(id_test_loader) * 100, TNR / len(ood_test_loader) * 100, threshold))
    print('ID_ACC:{:.4}% |OOD_ACC: {:.4}%'.format(ID_ACC / len(id_test_loader), OOD_ACC / len(ood_test_loader)))


def get_trained_features(model, train_dataloader, args):
    
    mean_list = []
    cov_list = []

    class_num = 10
   
    model = model.cuda()
    model.train()
    
    cov = torch.zeros(class_num, class_num, dtype=torch.float).cuda()

    with torch.no_grad():
        for i in range(class_num):
            feature_list = []

            for x, y in train_dataloader:
                x = x.cuda()
                y = y.cuda()

                pred, _ = model(x[y == i])
                feature_list.append(pred)

            feature_vector = torch.cat(feature_list, dim=0)
            
            data_num = feature_vector.size(0)

            mean = feature_vector.mean(dim=0)
            new_feature_vector = (feature_vector - mean[None, :])
            for row in new_feature_vector:
                row = row.unsqueeze(1)
                cov += torch.matmul(row, row.t())

            mean_list.append(mean)
            print(i)

        return mean_list, cov


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    def parse_args():
        import argparse
        parser = argparse.ArgumentParser('Mahalanobis-args')
        
        # experimental settings
        parser.add_argument('--seed', type=int, default=0, help='Random seed.')   
        parser.add_argument('--alg', type=str, default='mahalanobis', help='baseline | mahalanobis')

        parser.add_argument('--train_bs', type=int, default=1000, help='Batch size of in_trainloader.')
        parser.add_argument('--test_bs', type=int, default=1000, help='Batch size of in_testloader and out_testloader.')   
        parser.add_argument('--threshold', type=int, default=35, help='Threshold.')
        parser.add_argument('--num_workers', type=int, default=0)

        args = parser.parse_args()

        return args

    # arg parse
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
                                        download=False, transform=transform)
    id_trainloader = torch.utils.data.DataLoader(id_trainset, batch_size=args.train_bs,
                                            shuffle=False, num_workers=args.num_workers)

    # id_testloader will be used for test the given ood detection algorithm
    id_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
    id_testloader = torch.utils.data.DataLoader(id_testset, batch_size=args.test_bs,
                                            shuffle=False, num_workers=args.num_workers)
    
    """
    out-of-distribtuion data looader(SVHN)
    """

    # ood_testloader will be used for test the given ood detection algorithm
    ood_testset = torchvision.datasets.SVHN(root='./data', split='test',
                                        download=False, transform=transform)
    ood_testloader = torch.utils.data.DataLoader(ood_testset, batch_size=args.test_bs,
                                            shuffle=False, num_workers=args.num_workers)
    
    # load model trained on CIFAR-10 
    model = ResNet34()
    model.load_state_dict(torch.load('./model/resnet34-31.pth'))

    # ood dectection test
    if args.alg == 'baseline':
        print('result of baseline alg')
        ood_test_baseline(model, id_trainloader, id_testloader, ood_testloader, args)
    elif args.alg == 'mahalanobis':
        print('result of mahalanobis alg')
        ood_test_mahalanobis(model, id_trainloader, id_testloader, ood_testloader, args)
    else:
        print('--alg should be baseline or mahalanobis')