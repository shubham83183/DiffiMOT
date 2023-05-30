# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from ..util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from torch.utils.tensorboard import SummaryWriter



class HungarianMatcherDiff(torch.autograd.Function):
    """
    Torch module calculating the solution of the travelling salesman problem on a given distance matrix
    using a Gurobi implementation of a cutting plane algorithm.
    """

    @staticmethod
    def forward(ctx, *input):
        """
        distance_matrices: torch.Tensor of shape [batch_size, num_vertices, num_vertices]
        return: torch.Tenspr of shape [batch_size, num_vertices, num_vertices] 0-1 indicator matrices of the solution
        """
        ctx.cost_matrix = input[0].detach().cpu()#.numpy()
        ctx.sizes = input[1]
        ctx.writer1 = input[2]
        ctx.count = input[3]
        selected = [linear_sum_assignment(c)
                   for i, c in enumerate(input[0].split(input[1], -1))]
        cost_zeros = [np.zeros_like(c) for i, c in enumerate(input[0].split(input[1], -1))]
        solution = None
        for index, cost in zip(selected, cost_zeros):
            cost[index] = 1
            if solution is None:
                solution = cost
            else:
                solution = np.append(solution, cost, axis=1)
        ctx.solution = torch.tensor(solution)
        return torch.from_numpy(solution).float().to(input[0].device)

    @staticmethod
    def backward(ctx, *grad_output):
        assert grad_output[0].shape == ctx.solution.shape
        grad_output_numpy = grad_output[0].detach().cpu()#.numpy()

        #lambda_val = torch.abs(torch.mean(ctx.cost_matrix[ctx.cost_matrix != float("inf")])/torch.mean(grad_output_numpy))
        #lambda_val = torch.mean(ctx.cost_matrix[ctx.cost_matrix != float("inf")])/torch.mean(grad_output_numpy)
        lambda_val = torch.mean(torch.abs((ctx.cost_matrix[ctx.cost_matrix != float("inf")]) /torch.mean(grad_output_numpy)))

        ctx.count += 1
        if not ctx.count % 1000:
            ctx.writer1.add_scalar('lambda_val', lambda_val,global_step =ctx.count)

        if lambda_val <=1000:
            lambda_val = 1000
        elif lambda_val >=5000:
            lambda_val = 5000 
        
        if not ctx.count % 1000:
            ctx.writer1.add_scalar('lambda_valSat', lambda_val, global_step = ctx.count)


        cost_matrix_prime = ctx.cost_matrix + lambda_val * grad_output_numpy
        better_selected = [linear_sum_assignment(c)
                             for i, c in enumerate(cost_matrix_prime.split(ctx.sizes, -1))]
        modCost_zeros = [np.zeros_like(c) for i, c in enumerate(cost_matrix_prime.split(ctx.sizes, -1))]
        mod_solution = None
        for index, cost in zip(better_selected, modCost_zeros):
            cost[index]=1
            if mod_solution is None:
                mod_solution = cost
            else:
                mod_solution = np.append(mod_solution, cost, axis=1)
        gradient = -(ctx.solution - mod_solution) / lambda_val
        return gradient.to(grad_output[0].device), None, None, None



class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 focal_loss: bool = False, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """ Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates
                       in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the
                       matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.count = torch.tensor([0], dtype=int, requires_grad=False)
        self.solver = HungarianMatcherDiff()
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.writer1 = SummaryWriter("runs/crowd_private_detection_1000_5000_Lambda_values_Abs_mean")

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the
                                classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted
                               box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target
                     is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number
                           of ground-truth objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        #
        # [batch_size * num_queries, num_classes]
        if self.focal_loss:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        # [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        if self.focal_loss:
            neg_cost_class = (1 - self.focal_alpha) * (out_prob ** self.focal_gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.focal_alpha * ((1 - out_prob) ** self.focal_gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            # Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        cost_matrix = self.cost_bbox * cost_bbox \
            + self.cost_class * cost_class \
            + self.cost_giou * cost_giou
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        for i, target in enumerate(targets):
            if 'track_query_match_ids' not in target:
                continue

            prop_i = 0
            for j in range(cost_matrix.shape[1]):
                # if target['track_queries_fal_pos_mask'][j] or target['track_queries_placeholder_mask'][j]:
                if target['track_queries_fal_pos_mask'][j]:
                    # false positive and palceholder track queries should not
                    # be matched to any target
                    cost_matrix[i, j] = np.inf
                elif target['track_queries_mask'][j]:
                    track_query_id = target['track_query_match_ids'][prop_i]
                    prop_i += 1

                    cost_matrix[i, j] = np.inf
                    cost_matrix[i, :, track_query_id + sum(sizes[:i])] = np.inf
                    cost_matrix[i, j, track_query_id + sum(sizes[:i])] = -1
        cost_matrix = torch.hstack([c[i] for i, c in enumerate(cost_matrix.split(sizes, -1))])
        # self.count += 1
        out = self.solver.apply(cost_matrix, sizes,self.writer1, self.count)
        # indices = [linear_sum_assignment(c[i])
        #    for i, c in enumerate(cost_matrix.split(sizes, -1))]

        index = [np.nonzero(x) for i, x in enumerate(out.split(sizes,-1))]
        indices = [(y[:, 0], y[:, 1]) for i, y in enumerate(index)]
        #return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        #       for i, j in indices]
        return out.to(cost_class.device), sizes, [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
               for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,)
