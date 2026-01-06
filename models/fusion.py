import torch 
from packages.RatSPN.spn.rat_spn import make_rat
from packages.EinsumNet.simple_einet.einet import EinetConfig, Einet
from packages.EinsumNet.simple_einet.einet_mixture import EinetMixture
from packages.EinsumNet.simple_einet.layers.distributions.binomial import Binomial
from packages.EinsumNet.simple_einet.layers.distributions.categorical import Categorical, ConditionalCategorical
from packages.EinsumNet.simple_einet.layers.distributions.normal import RatNormal, Normal
from packages.EinsumNet.simple_einet.layers.distributions.dirichlet import Dirichlet
from packages.EinsumNet.simple_einet.layers.distributions.bernoulli import Bernoulli
from packages.EinsumNet.simple_einet.layers.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import numpy as np
import copy
from ensemble_boxes import *

class WeightedMean(torch.nn.Module):
    """Implements a module that returns performs weighted mean."""
    
    def __init__(self, weight_dims, normalize_dim, multilabel=False):
        super(WeightedMean, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(size=tuple(weight_dims)))
        self.normalize_dim = normalize_dim
        self.softmax = torch.nn.Softmax(dim=normalize_dim)
        self.multilabel = multilabel 
        
    def forward(self, x, context=None, **kwargs):
        x = x[:,:,:,0] if self.multilabel else x
        x = (x*self.softmax(self.weights)).sum(self.normalize_dim)
        return x

class NoisyOR(torch.nn.Module):
    """Implements a module that returns performs noisy or"""
    
    def __init__(self, normalize_dim, multilabel=False):
        super(NoisyOR, self).__init__()
        self.normalize_dim = normalize_dim
        self.multilabel = multilabel
        
    def forward(self, x, context=None, **kwargs):
        x = x[:,:,:,0] if self.multilabel else x
        y = 1 - torch.prod(1-x,dim=self.normalize_dim)
        y = y if self.multilabel else y/y.sum(dim=-1,keepdim=True)
        return y


class RatSPN(torch.nn.Module):
    """Implements a module that returns performs weighted mean."""
    
    def __init__(self, num_features, classes, leaves=10, sums=10, num_splits=10, dropout=0.0, multilabel=False):
        super(RatSPN, self).__init__()
        self.model = make_rat(num_features=num_features, classes=classes, leaves=leaves, sums=sums, num_splits=num_splits, dropout=dropout)
        
    def forward(self, x, context=None):
        return self.model(x).exp()
    
class EinsumNet(torch.nn.Module):
    """Implements a module that returns performs weighted mean."""
    
    def __init__(self, num_features, num_channels, depth, num_sums, num_leaves,
                 num_repetitions, num_classes, leaf_type, leaf_kwargs, einet_mixture=False,
                 layer_type='einsum', conditional_leaf=False, conditional_sum=False,
                 sum_cond_fn_type=None, sum_cond_fn_args=None, dropout=0.0, multilabel=False):
        super(EinsumNet, self).__init__()
        leaf_type, leaf_kwargs = eval(leaf_type), leaf_kwargs
        self.config = EinetConfig(
            num_features=num_features,
            num_channels=num_channels,
            depth=depth,
            num_sums=num_sums,
            num_leaves=num_leaves,
            num_repetitions=num_repetitions,
            num_classes=num_classes,
            leaf_kwargs=leaf_kwargs,
            leaf_type=leaf_type,
            dropout=dropout,
            layer_type=layer_type,
            conditional_leaf=conditional_leaf,
            conditional_sum=conditional_sum,
            sum_cond_fn_type=sum_cond_fn_type,
            sum_cond_fn_args={} if sum_cond_fn_args is None else sum_cond_fn_args,
        )
        self.multilabel = multilabel
        if einet_mixture:
            self.model = EinetMixture(n_components=num_classes, einet_config=self.config)
        else:
            self.model = Einet(self.config)
            
    def forward(self, x, context=None, marginalized_scopes=None):
        if self.multilabel:
            x = x.view(x.shape[0],-1,x.shape[3]) if len(x.shape)==4 else x.view(x.shape[0],-1) 
            probs = self.model(x.unsqueeze(1).unsqueeze(3).unsqueeze(3),cond_input=context,marginalized_scopes=marginalized_scopes).exp()
            probs = probs.view(probs.shape[0],-1,2)
            probs = probs/probs.sum(dim=-1, keepdim=True)
            return probs[:,:,1]
            # return probs
        else:    
            probs = self.model(x.unsqueeze(1).unsqueeze(3).unsqueeze(3),cond_input=context,marginalized_scopes=marginalized_scopes)
            probs = probs.exp()
            return probs/probs.sum(dim=-1,keepdim=True)
       
class FlowCircuit(torch.nn.Module):
    """Implements a module that returns performs weighted mean."""
    
    def __init__(self, num_features, classes, leaves=10, sums=10, num_splits=10, dropout=0.0):
        super(FlowCircuit, self).__init__()
        raise NotImplementedError
  
class CredibilityWeightedMean(EinsumNet):
    """Implements a module that returns performs weighted mean."""
    
    def forward(self, x, context=None, marginalized_scopes=None):
        if self.multilabel:
            x = x.view(x.shape[0],-1,x.shape[3]) if len(x.shape)==4 else x.view(x.shape[0],-1) 
            probs = self.model(x.unsqueeze(1).unsqueeze(3).unsqueeze(3),cond_input=context,marginalized_scopes=marginalized_scopes).exp()
            probs = probs.view(probs.shape[0],-1,2)
            probs = probs/probs.sum(dim=-1, keepdim=True)
            return probs[:,:,1]
            # return probs
        else:    
            probs = self.model(x.unsqueeze(1).unsqueeze(3).unsqueeze(3),cond_input=context,marginalized_scopes=marginalized_scopes).exp()
            probs = probs/probs.sum(dim=-1,keepdim=True)
            credibility = []
            for i in range(x.shape[1]):
                p_y_pi = self.model(x.unsqueeze(1).unsqueeze(3).unsqueeze(3),cond_input=context, marginalized_scopes=[i]).exp()
                p_y_pi = p_y_pi/p_y_pi.sum(dim=-1, keepdim=True)
                # credibility += [-JSD()(probs,p_y_pi).exp().sum(dim=-1).view(-1,1,1)]
                credibility += [-torch.nn.functional.kl_div(probs,p_y_pi).view(-1,1,1)]
            credibility = torch.cat(credibility,dim=1)
            credibility = credibility/credibility.sum(dim=1, keepdim=True)
            # self.loss = -probs.log()
            assert not credibility.isnan().any() 
            return (x*credibility).sum(dim=1)
        
class JSD(torch.nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor, eps=1e-12):
        p, q = p.clamp(eps), q.clamp(eps)
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(p.log(), m) + self.kl(q.log(), m))
    
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1

    label = torch.nn.functional.one_hot(p, num_classes=c)
    
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))









class TMC(torch.nn.Module):


    def __init__(self, classes, modalities, multilabel = False):
        """
        :param classes: Number of classification categories
        :param modalities: Number of views
        """
        super(TMC, self).__init__()
        self.num_classes = classes
        self.multilabel = multilabel
        self.num_modalities = modalities


    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.num_classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.num_classes, 1), b[1].view(-1, 1, self.num_classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.num_classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    

    def forward(self, x, context=None, **kwargs):
        x = torch.chunk(x, chunks= 6, dim=1)
        evidence = [out.squeeze(dim = 1) for out in x]
        alpha = [ev + 1 for ev in evidence]
        combin_alpha = self.DS_Combin(alpha)
        return combin_alpha
    





class HydraFusion(nn.Module):
    '''
    Represents the final fusion block in HydraFusion.
    Different fusion types currently implemented:
        1 = WBF, 2 = NMS, 3 = Soft-NMS
    '''
    def __init__(self, config, fusion_type, weights, iou_thr, skip_box_thr, sigma, alpha):
        super(HydraFusion, self).__init__()
        
        self.config = config
        self.weights = weights #output losses should be a list for losses for each branch
        # This weights variable can be tuned to favor certain branches over others
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.sigma = sigma
        self.fusion_type = fusion_type
        self.alpha = alpha

    ''' Function from the RADIATE SDK'''
    def transform(self, LidarToCamR, LidarToCamT):
        Rx = self.RX(LidarToCamR)
        Ry = self.RY(LidarToCamR)
        Rz = self.RZ(LidarToCamR)

        R = np.array([[1, 0, 0],
                      [0, 0, 1],
                      [0, -1, 0]]).astype(np.float)
        R = np.matmul(R, np.matmul(Rx, np.matmul(Ry, Rz)))

        LidarToCam = np.array([[R[0, 0], R[0, 1], R[0, 2], 0.0],
                               [R[1, 0], R[1, 1], R[1, 2], 0.0],
                               [R[2, 0], R[2, 1], R[2, 2], 0.0],
                               [LidarToCamT[0], LidarToCamT[1], LidarToCamT[2], 1.0]]).T
        return LidarToCam

    ''' Function from the RADIATE SDK'''
    def RX(self, LidarToCamR):
        thetaX = np.deg2rad(LidarToCamR[0])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(thetaX), -np.sin(thetaX)],
                       [0, np.sin(thetaX), np.cos(thetaX)]]).astype(np.float)
        return Rx

    ''' Function from the RADIATE SDK'''
    def RY(self, LidarToCamR):
        thetaY = np.deg2rad(LidarToCamR[1])
        Ry = np.array([[np.cos(thetaY), 0, np.sin(thetaY)],
                       [0, 1, 0],
                       [-np.sin(thetaY), 0, np.cos(thetaY)]])
        return Ry

    ''' Function from the RADIATE SDK'''
    def RZ(self, LidarToCamR):
        thetaZ = np.deg2rad(LidarToCamR[2])
        Rz = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0],
                       [np.sin(thetaZ), np.cos(thetaZ), 0],
                       [0, 0, 1]]).astype(np.float)
        return Rz
    
    ''' Function from the RADIATE SDK'''
    """ 
    method to project the bounding boxes to the camera
        :param annotations: the annotations for the current frame
        :type annotations: list
        :param intrinsict: intrisic camera parameters
        :type intrinsict: np.array
        :param extrinsic: extrinsic parameters
        :type extrinsic: np.array
        :return: dictionary with the list of bbounding boxes with camera coordinate frames
        :rtype: dict
    """
    def project_bboxes_to_camera(self, annotations, labels, intrinsict, extrinsic):
        
        bboxes_3d = []
        heights = {'car': 1.5,
                        'bus': 3,
                        'truck': 2.5,
                        'pedestrian': 1.8,
                        'van': 2,
                        'group_of_pedestrians': 1.8,
                        'motorbike': 1.5,
                        'bicycle': 1.5,
                        'vehicle': 1.5
                        }
        class_names=['background', 'car', 'van', 'truck', 'bus', 'motorbike', 'bicycle','pedestrian','group_of_pedestrians']
        ii = 0
        for object in annotations:
            obj = {}
            bb = np.float64(object.cpu())
            idx = np.int(labels[ii].cpu())
            height = heights[class_names[idx]]
            #converts [x1,y1,x2,y2] to [x1,y1,w,h]:
            bb = np.array([ bb[0], bb[1], (bb[2]-bb[0]), (bb[3]-bb[1]) ])
            rotation = 0
            bbox_3d = self.__get_projected_bbox(bb, rotation, intrinsict, extrinsic, height)
            obj['bbox_3d'] = bbox_3d
            bboxes_3d.append(obj)
            ii = ii + 1

        return bboxes_3d

    ''' Function from the RADIATE SDK'''
    def __get_projected_bbox(self, bb, rotation, cameraMatrix, extrinsic, obj_height=2):
        """get the projected boundinb box to some camera sensor
        """
        rotation = np.deg2rad(-rotation)
        res = 0.173611 #self.config['radar_calib']['range_res']
        cx = bb[0] + bb[2] / 2
        cy = bb[1] + bb[3] / 2
        T = np.array([[cx], [cy]])
        pc = 0.2
        bb = [bb[0]+bb[2]*pc, bb[1]+bb[3]*pc, bb[2]-bb[2]*pc, bb[3]-bb[3]*pc]

        R = np.array([[np.cos(rotation), -np.sin(rotation)],
                      [np.sin(rotation), np.cos(rotation)]])

        points = np.array([[bb[0], bb[1]],
                           [bb[0] + bb[2], bb[1]],
                           [bb[0] + bb[2], bb[1] + bb[3]],
                           [bb[0], bb[1] + bb[3]],
                           [bb[0], bb[1]],
                           [bb[0] + bb[2], bb[1] + bb[3]]]).T

        points = points - T
        points = np.matmul(R, points) + T
        points = points.T

        points[:, 0] = points[:, 0] - 576 #self.config['radar_calib']['range_cells']
        points[:, 1] = 576 - points[:, 1] #self.config['radar_calib']['range_cells'] - points[:, 1]
        points = points * res

        points = np.append(points, np.ones(
            (points.shape[0], 1)) * -1.7, axis=1)
        p1 = points[0, :]
        p2 = points[1, :]
        p3 = points[2, :]
        p4 = points[3, :]

        p5 = np.array([p1[0], p1[1], p1[2] + obj_height])
        p6 = np.array([p2[0], p2[1], p2[2] + obj_height])
        p7 = np.array([p3[0], p3[1], p3[2] + obj_height])
        p8 = np.array([p4[0], p4[1], p4[2] + obj_height])
        points = np.array([p1, p2, p3, p4, p1, p5, p6, p2, p6,
                           p7, p3, p7, p8, p4, p8, p5, p4, p3, p2, p6, p3, p1])

        points = np.matmul(np.append(points, np.ones(
            (points.shape[0], 1)), axis=1), extrinsic.T)

        points = (points / points[:, 3, None])[:, 0:3]

        filtered_indices = []
        for i in range(points.shape[0]):
            #if (points[i, 2] > 0 and points[i, 2] < self.config['max_range_bbox_camera']):
            if (points[i, 2] > 0 and points[i, 2] < 100):
                filtered_indices.append(i)

        points = points[filtered_indices]

        fx = cameraMatrix[0, 0]
        fy = cameraMatrix[1, 1]
        cx = cameraMatrix[0, 2]
        cy = cameraMatrix[1, 2]

        xIm = np.round((fx * points[:, 0] / points[:, 2]) + cx).astype(np.int)
        yIm = np.round((fy * points[:, 1] / points[:, 2]) + cy).astype(np.int)

        proj_bbox_3d = []
        for ii in range(1, xIm.shape[0]):
            proj_bbox_3d.append([xIm[ii], yIm[ii]])
        proj_bbox_3d = np.array(proj_bbox_3d)
        return proj_bbox_3d


    def forward(self, output_losses, output_detections, fusion_sweep=False):
        #init some parameters:
        ylim = 376; xlim = 672
        fxl = 3.379191448899105e+02; fyl=  3.386957068549526e+02
        fxr = 337.873451599077 ; fyr = 338.530902554779
        cxl =  3.417366010946575e+02; cyl= 2.007359735313929e+02
        cxr = 329.137695760749 ; cyr = 186.166590759716
        left_cam_mat = np.array([[fxl, 0, cxl],
                                    [0, fyl, cyl],
                                    [0,  0,  1]])
        right_cam_mat = np.array([[fxr, 0, cxr],
                                    [0, fyr, cyr],
                                    [0,  0,  1]])

        RadarT = np.array([0.0, 0.0, 0.0])
        RadarR = np.array([0.0, 0.0, 0.0])
        LidarT = np.array([0.6003, -0.120102, 0.250012])
        LidarR = np.array([0.0001655, 0.000213, 0.000934])
        LeftT = np.array([0.34001, -0.06988923, 0.287893])
        LeftR = np.array([1.278946, -0.530201, 0.000132])
        RightT = np.array([0.4593822, -0.0600343, 0.287433309324])
        RightR = np.array([0.8493049332, 0.37113944, 0.000076230])

        RadarToLeftT = RadarT - LeftT; RadarToRightT = RadarT - RightT
        RadarToLeftR = RadarR - LeftR; RadarToRightR = RadarR - RightR
        RadarToLeft = self.transform(RadarToLeftR, RadarToLeftT)
        RadarToRight = self.transform(RadarToRightR, RadarToRightT)

        LidarToLeftT = LidarT - LeftT; LidarToRightT = LidarT - RightT
        LidarToLeftR = LidarR - LeftR; LidarToRightR = LidarR - RightR
        LidarToLeft = self.transform(LidarToLeftR, LidarToLeftT)
        LidarToRight = self.transform(LidarToRightR, LidarToRightT)

        num_branches = len(output_detections) #get the number of branches
        good_branches = [] #will return a list of good branches with detections
        bboxes_3d = {} ; bboxes_3dl = {}
        output_detections_copy = copy.deepcopy(output_detections)
        for i in output_detections_copy:
            # Step 1: Handle case where branches have no detections: moved to later to handle radar,lidar,radar_lidar out of range issues
            if output_detections_copy[i][0]['boxes'].numel(): 
                good_branches.append(i)
            
            if i=='radar': #radar
                #call convert radar to camera bbox function on output_detections[i][0]['boxes']
                bboxes_3d = self.project_bboxes_to_camera(output_detections_copy[i][0]['boxes'], output_detections_copy[i][0]['labels'],right_cam_mat, RadarToRight)
                j = 0
                for k in range(len(bboxes_3d)):
                    #Added fix for empty boxes that were out of range of camera during transformation
                    if not(bboxes_3d[k]['bbox_3d'].any()):
                        output_detections_copy[i][0]['boxes'] = torch.cat([output_detections_copy[i][0]['boxes'][0:k-j], output_detections_copy[i][0]['boxes'][k-j+1:]])
                        output_detections_copy[i][0]['labels'] = torch.cat([output_detections_copy[i][0]['labels'][0:k-j], output_detections_copy[i][0]['labels'][k-j+1:]])
                        output_detections_copy[i][0]['scores'] = torch.cat([output_detections_copy[i][0]['scores'][0:k-j], output_detections_copy[i][0]['scores'][k-j+1:]])
                        j = j + 1
                        continue
                    x1 = np.min(bboxes_3d[k]['bbox_3d'][:,0]); x2 = np.max(bboxes_3d[k]['bbox_3d'][:,0]) 
                    y1 = np.min(bboxes_3d[k]['bbox_3d'][:,1]); y2 = np.max(bboxes_3d[k]['bbox_3d'][:,1])
                    #adding a fix for boxes with zero width or height caused by world-to-pixel transformation rounding
                    if (x1==x2):
                        x2 = x2+1
                    if (y2==y1):
                        y2 = y2+1
                    if (0<x1<672) & (0<x2<672) & (0<y1<376) & (0<y2<376):
                        output_detections_copy[i][0]['boxes'][k-j] = torch.tensor([x1,y1,x2,y2], dtype=torch.float32)
                    else:
                        #remove all entries from the dict or as an alternative, set score =0 so not used in fusion
                        #output_detections[i][0]['scores'][k] = 0
                        output_detections_copy[i][0]['boxes'] = torch.cat([output_detections_copy[i][0]['boxes'][0:k-j], output_detections_copy[i][0]['boxes'][k-j+1:]])
                        output_detections_copy[i][0]['labels'] = torch.cat([output_detections_copy[i][0]['labels'][0:k-j], output_detections_copy[i][0]['labels'][k-j+1:]])
                        output_detections_copy[i][0]['scores'] = torch.cat([output_detections_copy[i][0]['scores'][0:k-j], output_detections_copy[i][0]['scores'][k-j+1:]])
                        j = j + 1
                        continue

            if i=='lidar': #lidar
                #call convert radar to camera bbox function on output_detections[i][0]['boxes']
                bboxes_3dl = self.project_bboxes_to_camera(output_detections_copy[i][0]['boxes'],output_detections_copy[i][0]['labels'],right_cam_mat, LidarToRight)
                j = 0
                for k in range(len(bboxes_3dl)):
                    if not(bboxes_3dl[k]['bbox_3d'].any()):
                        output_detections_copy[i][0]['boxes'] = torch.cat([output_detections_copy[i][0]['boxes'][0:k-j], output_detections_copy[i][0]['boxes'][k-j+1:]])
                        output_detections_copy[i][0]['labels'] = torch.cat([output_detections_copy[i][0]['labels'][0:k-j], output_detections_copy[i][0]['labels'][k-j+1:]])
                        output_detections_copy[i][0]['scores'] = torch.cat([output_detections_copy[i][0]['scores'][0:k-j], output_detections_copy[i][0]['scores'][k-j+1:]])
                        j = j + 1
                        continue
                    x1 = np.min(bboxes_3dl[k]['bbox_3d'][:,0]); x2 = np.max(bboxes_3dl[k]['bbox_3d'][:,0]) 
                    y1 = np.min(bboxes_3dl[k]['bbox_3d'][:,1]); y2 = np.max(bboxes_3dl[k]['bbox_3d'][:,1])
                    #adding a fix for boxes with zero width or height caused by world-to-pixel transformation rounding
                    if (x1==x2):
                        x2 = x2+1
                    if (y2==y1):
                        y2 = y2+1
                    if (0<x1<672) & (0<x2<672) & (0<y1<376) & (0<y2<376):
                        output_detections_copy[i][0]['boxes'][k-j] = torch.tensor([x1,y1,x2,y2], dtype=torch.float32)
                    else:
                        #remove all entries from the dict or as an alternative, set score =0 so not used in fusion
                        #output_detections[i][0]['scores'][k] = 0
                        output_detections_copy[i][0]['boxes'] = torch.cat([output_detections_copy[i][0]['boxes'][0:k-j], output_detections_copy[i][0]['boxes'][k-j+1:]])
                        output_detections_copy[i][0]['labels'] = torch.cat([output_detections_copy[i][0]['labels'][0:k-j], output_detections_copy[i][0]['labels'][k-j+1:]])
                        output_detections_copy[i][0]['scores'] = torch.cat([output_detections_copy[i][0]['scores'][0:k-j], output_detections_copy[i][0]['scores'][k-j+1:]])
                        j = j + 1
                        continue
            
            if i=='radar_lidar': #lidar
                #call convert radar to camera bbox function on output_detections[i][0]['boxes']
                bboxes_3dlr = self.project_bboxes_to_camera(output_detections_copy[i][0]['boxes'], output_detections_copy[i][0]['labels'],right_cam_mat, RadarToRight)
                j = 0
                for k in range(len(bboxes_3dlr)):
                    if not(bboxes_3dlr[k]['bbox_3d'].any()):
                        output_detections_copy[i][0]['boxes'] = torch.cat([output_detections_copy[i][0]['boxes'][0:k-j], output_detections_copy[i][0]['boxes'][k-j+1:]])
                        output_detections_copy[i][0]['labels'] = torch.cat([output_detections_copy[i][0]['labels'][0:k-j], output_detections_copy[i][0]['labels'][k-j+1:]])
                        output_detections_copy[i][0]['scores'] = torch.cat([output_detections_copy[i][0]['scores'][0:k-j], output_detections_copy[i][0]['scores'][k-j+1:]])
                        j = j + 1
                        continue
                    x1 = np.min(bboxes_3dlr[k]['bbox_3d'][:,0]); x2 = np.max(bboxes_3dlr[k]['bbox_3d'][:,0]) 
                    y1 = np.min(bboxes_3dlr[k]['bbox_3d'][:,1]); y2 = np.max(bboxes_3dlr[k]['bbox_3d'][:,1])
                    #adding a fix for boxes with zero width or height caused by world-to-pixel transformation rounding
                    if (x1==x2):
                        x2 = x2+1
                    if (y2==y1):
                        y2 = y2+1
                    if (0<x1<672) & (0<x2<672) & (0<y1<376) & (0<y2<376):
                        output_detections_copy[i][0]['boxes'][k-j] = torch.tensor([x1,y1,x2,y2], dtype=torch.float32)
                    else:
                        #remove all entries from the dict or as an alternative, set score =0 so not used in fusion
                        #output_detections[i][0]['scores'][k] = 0
                        output_detections_copy[i][0]['boxes'] = torch.cat([output_detections_copy[i][0]['boxes'][0:k-j], output_detections_copy[i][0]['boxes'][k-j+1:]])
                        output_detections_copy[i][0]['labels'] = torch.cat([output_detections_copy[i][0]['labels'][0:k-j], output_detections_copy[i][0]['labels'][k-j+1:]])
                        output_detections_copy[i][0]['scores'] = torch.cat([output_detections_copy[i][0]['scores'][0:k-j], output_detections_copy[i][0]['scores'][k-j+1:]])
                        j = j + 1
                        continue

        # Step 3: only use good detections
        good_branches2 = []
        for i in output_detections_copy:
            if output_detections_copy[i][0]['boxes'].numel():
                good_branches2.append(i)    
        good_detections = []
        for i in good_branches2:
            good_detections.append(output_detections_copy[i])
        

        # Step 4: Normalize the bbox pixel values for fusion
        #boxes_list should be a list of floats example for two branches (branch one has two objects, branch one has one):
        # Example [[ [0.00, 0.51, 0.81, 0.91], [0.10, 0.31, 0.71, 0.61], ], [ [0.04, 0.56, 0.84, 0.92],]]
        # scores_list # Example: [[0.9, 0.8], [0.3]]
        # labels_list  # Example: [[0, 1], [1]]

        #Size of the image: defined above: xlim,ylim
        
        boxes_list = []; scores_list = []; labels_list = []
        for i in good_detections:
            branch_boxes = []; branch_scores = []; branch_labels = []
            for j in np.float64(i[0]['boxes'].cpu()):
                if j[3] > ylim: j[3] = round(j[3]) #handles case where j[3] = 376.0001
                if j[2] > xlim: j[2] = round(j[2])
                branch_boxes.append([ j[0]/xlim , j[1]/ylim , j[2]/xlim , j[3]/ylim  ])

            boxes_list.append(branch_boxes)
            scores_list.append(i[0]['scores'].cpu().numpy().tolist())
            labels_list.append(i[0]['labels'].cpu().numpy().tolist())


        if not(fusion_sweep):
            if not(bool(boxes_list)): #checks if there are any bounding boxes
                fboxes1, fscores1, flabels1 = np.empty([0,4]), np.array([]),np.array([]) 
                fboxes2, fscores2, flabels2 = np.empty([0,4]), np.array([]),np.array([]) 
                fboxes3, fscores3, flabels3 = np.empty([0,4]), np.array([]),np.array([])
            elif not(bool(boxes_list[0])):
                fboxes1, fscores1, flabels1 = np.empty([0,4]), np.array([]),np.array([]) 
                fboxes2, fscores2, flabels2 = np.empty([0,4]), np.array([]),np.array([]) 
                fboxes3, fscores3, flabels3 = np.empty([0,4]), np.array([]),np.array([])
            else:
                fboxes1, fscores1, flabels1 = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=len(boxes_list)*[1], iou_thr=self.iou_thr, skip_box_thr=self.skip_box_thr)
                fboxes2, fscores2, flabels2 = nms(boxes_list, scores_list, labels_list, weights=len(boxes_list)*[1], iou_thr=self.iou_thr)
                fboxes3, fscores3, flabels3 = soft_nms(boxes_list, scores_list, labels_list, weights=len(boxes_list)*[1], iou_thr=self.iou_thr, sigma=self.sigma, thresh=self.skip_box_thr) 

            # Step 5b: rescale up the predictions
            ffboxes1 = []
            for i in fboxes1:
                ffboxes1.append([i[0]*xlim , i[1]*ylim , i[2]*xlim , i[3]*ylim ])
            ffboxes2 = []
            for i in fboxes2:
                ffboxes2.append([i[0]*xlim , i[1]*ylim , i[2]*xlim , i[3]*ylim ])
            ffboxes3 = []
            for i in fboxes3:
                ffboxes3.append([i[0]*xlim , i[1]*ylim , i[2]*xlim , i[3]*ylim ])

            # Step 6: Compile the results
            output_detections.update({'fused1':[{'boxes': torch.tensor(ffboxes1, device=self.config.device),'labels': torch.from_numpy(flabels1).to(self.config.device),'scores': torch.from_numpy(fscores1).to(self.config.device) }]})
            output_detections.update({'fused2':[{'boxes': torch.tensor(ffboxes2, device=self.config.device),'labels': torch.from_numpy(flabels2).to(self.config.device),'scores': torch.from_numpy(fscores2).to(self.config.device) }]})
            output_detections.update({'fused3':[{'boxes': torch.tensor(ffboxes3, device=self.config.device),'labels': torch.from_numpy(flabels3).to(self.config.device),'scores': torch.from_numpy(fscores3).to(self.config.device) }]})
        else: #fusion sweep
            #current default: iou_thr=0.4, skip_box_thr=0.01, sigma=0.5
            iou_thr_range = [0.4,0.5,0.6] 
            skip_box_thr_range  = [0.01 , 0.1 , 0.3]
            sigma_range = [0.1 , 0.25, 0.5]
            sweep_num = len(iou_thr_range)*len(skip_box_thr_range)*len(sigma_range)
            fff = 0
            for iou in iou_thr_range:
                for skip_b in skip_box_thr_range:
                    for sig in sigma_range: 
            
                        if not(bool(boxes_list)): #checks if there are any bounding boxes
                            fboxes1, fscores1, flabels1 = np.empty([0,4]), np.array([]),np.array([]) 
                            fboxes2, fscores2, flabels2 = np.empty([0,4]), np.array([]),np.array([]) 
                            fboxes3, fscores3, flabels3 = np.empty([0,4]), np.array([]),np.array([])
                        elif not(bool(boxes_list[0])):
                            fboxes1, fscores1, flabels1 = np.empty([0,4]), np.array([]),np.array([]) 
                            fboxes2, fscores2, flabels2 = np.empty([0,4]), np.array([]),np.array([]) 
                            fboxes3, fscores3, flabels3 = np.empty([0,4]), np.array([]),np.array([])
                        else:
                            fboxes1, fscores1, flabels1 = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=len(boxes_list)*[1], iou_thr=iou, skip_box_thr=skip_b)
                            fboxes2, fscores2, flabels2 = nms(boxes_list, scores_list, labels_list, weights=len(boxes_list)*[1], iou_thr=iou)
                            fboxes3, fscores3, flabels3 = soft_nms(boxes_list, scores_list, labels_list, weights=len(boxes_list)*[1], iou_thr=iou, sigma=sig, thresh=skip_b) 

                        # Step 5b: rescale up the predictions
                        ffboxes1 = []
                        for i in fboxes1:
                            ffboxes1.append([i[0]*xlim , i[1]*ylim , i[2]*xlim , i[3]*ylim ])
                        ffboxes2 = []
                        for i in fboxes2:
                            ffboxes2.append([i[0]*xlim , i[1]*ylim , i[2]*xlim , i[3]*ylim ])
                        ffboxes3 = []
                        for i in fboxes3:
                            ffboxes3.append([i[0]*xlim , i[1]*ylim , i[2]*xlim , i[3]*ylim ])

                        # Step 6: Compile the results
                        fusion_key1 = 'fused1_' + 'iou_' + str(iou) +'_skip_' + str(skip_b)
                        fusion_key2 = 'fused2_' + 'iou_' + str(iou) 
                        fusion_key3 = 'fused3_' + 'iou_' + str(iou) +'_sig_'+ str(sig) + '_skip_' + str(skip_b)
                        output_detections.update({fusion_key1:[{'boxes': torch.tensor(ffboxes1, device=self.config.device),'labels': torch.from_numpy(flabels1).to(self.config.device),'scores': torch.from_numpy(fscores1).to(self.config.device) }]})
                        output_detections.update({fusion_key2:[{'boxes': torch.tensor(ffboxes2, device=self.config.device),'labels': torch.from_numpy(flabels2).to(self.config.device),'scores': torch.from_numpy(fscores2).to(self.config.device) }]})
                        output_detections.update({fusion_key3:[{'boxes': torch.tensor(ffboxes3, device=self.config.device),'labels': torch.from_numpy(flabels3).to(self.config.device),'scores': torch.from_numpy(fscores3).to(self.config.device) }]})
                        fff = fff + 1
        

        # Step 7: Compute the fused loss
        #TODO: calculate final loss
        final_loss = output_losses
        return final_loss, output_detections