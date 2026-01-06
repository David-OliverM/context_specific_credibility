import torch


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, **kwargs):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        # logging.info("using gamma={}".format(gamma))

    def forward(self, input, target):

        target = target.view(-1,1)

        # logpt = torch.nn.functional.log_softmax(input, dim=1) #since probabilities are being passed not logits
        logpt = torch.log(input) 
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        
        return loss.mean()


class MDCA(torch.nn.Module):
    def __init__(self):
        super(MDCA,self).__init__()

    def forward(self , output, target):
        # output = torch.softmax(output, dim=1) #since probabilities are being passed not logits
        # [batch, classes]
        loss = torch.tensor(0.0).cuda()
        batch, classes = output.shape
        for c in range(classes):
            avg_count = (target == c).float().mean()
            avg_conf = torch.mean(output[:,c])
            loss += torch.abs(avg_conf - avg_count)
        denom = classes
        loss /= denom
        return loss
    

class FLandMDCA(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, gamma=1.0, **kwargs):
        super(FLandMDCA, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.classification_loss = FocalLoss(gamma=self.gamma)
        self.MDCA = MDCA()

    def forward(self, probs, targets):
        loss_cls = self.classification_loss(probs, targets)
        loss_cal = self.MDCA(probs, targets)
        return loss_cls + self.beta * loss_cal