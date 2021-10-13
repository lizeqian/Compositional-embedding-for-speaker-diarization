from pyannote.audio.pipeline.speech_turn_assignment import SpeechTurnClosestAssignment
import torch
import itertools
import torch.nn.functional as F
from pyannote.pipeline.parameter import Uniform
from pyannote.core.utils.distance import cdist
from pyannote.core.utils.distance import dist_range
from pyannote.core.utils.distance import l2_normalize
from pyannote.core import Annotation
from pyannote.audio.pipeline.utils import assert_int_labels, assert_string_labels
from pyannote.audio.features.wrapper import Wrapper, Wrappable
from typing import Optional
import numpy as np
import warnings
from pyannote.pipeline import Pipeline

class ClosestAssignment(Pipeline):
    """Assign each sample to the closest target

    Parameters
    ----------
    metric : `str`, optional
        Distance metric. Defaults to 'cosine'
    normalize : `bool`, optional
        L2 normalize vectors before clustering.

    Hyper-parameters
    ----------------
    threshold : `float`
        Do not assign if distance greater than `threshold`.
    """

    def __init__(self, metric: Optional[str] = 'cosine',
                       normalize: Optional[bool] = False):

        super().__init__()
        self.metric = metric
        self.normalize = normalize

        min_dist, max_dist = dist_range(metric=self.metric,
                                        normalize=self.normalize)
        if not np.isfinite(max_dist):
            # this is arbitray and might lead to suboptimal results
            max_dist = 1e6
            msg = (f'bounding distance threshold to {max_dist:g}: '
                   f'this might lead to suboptimal results.')
            warnings.warn(msg)
        self.threshold = Uniform(min_dist, max_dist)

    def __call__(self, X_target, X):
        """Assign each sample to its closest class (if close enough)

        Parameters
        ----------
        X_target : `np.ndarray`
            (n_targets, n_dimensions) target embeddings
        X : `np.ndarray`
            (n_samples, n_dimensions) sample embeddings

        Returns
        -------
        assignments : `np.ndarray`
            (n_samples, ) sample assignments
        """

        if self.normalize:
            X_target = l2_normalize(X_target)
            X = l2_normalize(X)

        distance = cdist(X_target, X, metric=self.metric)
        idx = np.argsort(distance, axis=0)

        for i, k in enumerate(idx[0]):
            if distance[k, i] > self.threshold:
                # do not assign
                idx[0][i] = -i

        return idx

class ClosestAssignmentAlwaysAssign(ClosestAssignment):
    def __call__(self, X_target, X):
        if self.normalize:
            X_target = l2_normalize(X_target)
            X = l2_normalize(X)

        distance = cdist(X_target, X, metric=self.metric)
        idx = np.argsort(distance, axis=0)
        return idx

class SpeechTurnClosestAssignmentNew(SpeechTurnClosestAssignment):
    def __init__(self, embedding: Wrappable = None, metric: Optional[str] = "cosine"):
        super().__init__(embedding=embedding, metric=metric)
        self.closest_assignment = ClosestAssignment(metric=self.metric)

    def __call__(
        self, current_file: dict, speech_turns: Annotation, targets: Annotation
    ) -> Annotation:
        """Assign each speech turn to closest target (if close enough)

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.
        speech_turns : `Annotation`
            Speech turns. Should only contain `int` labels.
        targets : `Annotation`
            Targets. Should only contain `str` labels.

        Returns
        -------
        assigned : `Annotation`
            Assigned speech turns.
        """

        assert_string_labels(targets, "targets")
        assert_int_labels(speech_turns, "speech_turns")

        embedding = self._embedding(current_file)

        # gather targets embedding
        labels = targets.labels()
        X_targets, targets_labels = [], []
        for l, label in enumerate(labels):

            timeline = targets.label_timeline(label, copy=False)

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ["strict", "center", "loose"]:
                x = embedding.crop(timeline, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we don't have any embedding for it
            if len(x) < 1:
                continue

            targets_labels.append(label)
            X_targets.append(np.mean(x, axis=0))

        # gather speech turns embedding
        labels = speech_turns.labels()
        X, assigned_labels, skipped_labels = [], [], []
        for l, label in enumerate(labels):

            timeline = speech_turns.label_timeline(label, copy=False)

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ["strict", "center", "loose"]:
                x = embedding.crop(timeline, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we don't have any embedding for it
            if len(x) < 1:
                skipped_labels.append(label)
                continue

            assigned_labels.append(label)
            X.append(np.mean(x, axis=0))

        # assign speech turns to closest class
        assignments = self.closest_assignment(np.vstack(X_targets), np.vstack(X))
        mapping = {
            label: targets_labels[k]
            for label, k in zip(assigned_labels, assignments[0])
            if not k < 0
        }
        mapping1 = {
            label: targets_labels[k]
            for label, k in zip(assigned_labels, assignments[1])
            if not k < 0
        }
        return speech_turns.rename_labels(mapping=mapping), speech_turns.copy().rename_labels(mapping=mapping1)

class SpeechTurnClosestAssignmentMultiSpeaker(SpeechTurnClosestAssignment):
    def __init__(self, gnet, device, embedding: Wrappable = None, metric: Optional[str] = "cosine"):
        super(SpeechTurnClosestAssignmentMultiSpeaker, self).__init__(embedding, metric)
        self.closest_assignment = ClosestAssignmentAlwaysAssign(metric=self.metric)
        self.g_net = gnet.to(device)
        self.device = device

    def __call__(
        self, current_file: dict, speech_turns: Annotation, targets: Annotation
    ) -> Annotation:

        assert_string_labels(targets, "targets")
        assert_int_labels(speech_turns, "speech_turns")

        embedding = self._embedding(current_file)

        # gather targets embedding
        labels = targets.labels()
        X_targets, targets_labels = [], []
        for l, label in enumerate(labels):

            timeline = targets.label_timeline(label, copy=False)

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ["strict", "center", "loose"]:
                x = embedding.crop(timeline, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we don't have any embedding for it
            if len(x) < 1:
                continue

            targets_labels.append(label)
            X_targets.append(np.mean(x, axis=0))

        # gather speech turns embedding
        labels = speech_turns.labels()
        X, assigned_labels, skipped_labels = [], [], []
        for l, label in enumerate(labels):

            timeline = speech_turns.label_timeline(label, copy=False)

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ["strict", "center", "loose"]:
                x = embedding.crop(timeline, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we don't have any embedding for it
            if len(x) < 1:
                skipped_labels.append(label)
                continue

            assigned_labels.append(label)
            X.append(np.mean(x, axis=0))

        # assign speech turns to closest class
        targets = np.vstack(X_targets)
        num_targets = len(targets)
        targets_tensor = torch.tensor(targets).to(self.device).float()
        if targets_tensor.size(0) > 1:
        # targets_tensor = F.normalize(torch.tensor(targets).to(device).float())
            combinations = torch.tensor(list(itertools.combinations(list(range(num_targets)),2)))
            comb2_a = targets_tensor[combinations.transpose(-2, -1)[0]]
            comb2_b = targets_tensor[combinations.transpose(-2, -1)[1]]
            merged2 = self.g_net(comb2_a, comb2_b)
            new_targets = torch.cat([targets_tensor, merged2], 0).cpu().detach().numpy()
        else:
            new_targets = targets

        for comb in list(itertools.combinations(list(range(num_targets)),2)):
            targets_labels.append(f'{targets_labels[comb[0]]}_{targets_labels[comb[1]]}')
        assignments = self.closest_assignment(new_targets, np.vstack(X))
        mapping = {
            label: targets_labels[k]
            for label, k in zip(assigned_labels, assignments[0])
            if not k < 0
        }
        return speech_turns.rename_labels(mapping=mapping)

class SpeechTurnClosestAssignmentMerge(SpeechTurnClosestAssignment):
    def __init__(self, gnet, device, embedding: Wrappable = None, metric: Optional[str] = "cosine"):
        super(SpeechTurnClosestAssignmentMerge, self).__init__(embedding, metric)
        self.g_net = gnet.to(device)
        self.device = device
        self.closest_assignment = ClosestAssignmentAlwaysAssign(metric=self.metric)

    def __call__(
        self, current_file: dict, speech_turns: Annotation, targets: Annotation
    ) -> Annotation:
        assert_string_labels(targets, "targets")
        assert_int_labels(speech_turns, "speech_turns")

        embedding = self._embedding(current_file)

        # gather targets embedding
        labels = targets.labels()
        X_targets, targets_labels = [], []
        for l, label in enumerate(labels):

            timeline = targets.label_timeline(label, copy=False)

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ["strict", "center", "loose"]:
                x = embedding.crop(timeline, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we don't have any embedding for it
            if len(x) < 1:
                continue

            targets_labels.append(label)
            X_targets.append(np.mean(x, axis=0))

        # gather speech turns embedding
        labels = speech_turns.labels()
        X, assigned_labels, skipped_labels = [], [], []
        for l, label in enumerate(labels):

            timeline = speech_turns.label_timeline(label, copy=False)

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ["strict", "center", "loose"]:
                x = embedding.crop(timeline, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we don't have any embedding for it
            if len(x) < 1:
                skipped_labels.append(label)
                continue

            assigned_labels.append(label)
            X.append(np.mean(x, axis=0))

        assignments_original = self.closest_assignment(np.vstack(X_targets), np.vstack(X))
        mapping_original = {
            label: targets_labels[k]
            for label, k in zip(assigned_labels, assignments_original[0])
            if not k < 0
        }

        speech_turn_original = speech_turns.copy().rename_labels(mapping=mapping_original)

        # assign speech turns to closest class
        targets = np.vstack(X_targets)
        num_targets = len(targets)
        targets_tensor = torch.tensor(targets).to(self.device).float()
        # targets_tensor = F.normalize(torch.tensor(targets).to(device).float())
        if num_targets > 1:
            combinations = torch.tensor(list(itertools.combinations(list(range(num_targets)),2)))
            comb2_a = targets_tensor[combinations.transpose(-2, -1)[0]]
            comb2_b = targets_tensor[combinations.transpose(-2, -1)[1]]
            merged2 = self.g_net(comb2_a, comb2_b)
            new_targets = merged2.cpu().detach().numpy()
            targets_labels_merge = []
            for comb in list(itertools.combinations(list(range(num_targets)),2)):
                targets_labels_merge.append(f'{targets_labels[comb[0]]}_{targets_labels[comb[1]]}')
        else:
            new_targets = targets
            targets_labels_merge = [f'{targets_labels[0]}_{targets_labels[0]}']
        assignments = self.closest_assignment(new_targets, np.vstack(X))
        mapping = {
            label: targets_labels_merge[k]
            for label, k in zip(assigned_labels, assignments[0])
            if not k < 0
        }
        speech_turn_merge = speech_turns.rename_labels(mapping=mapping)

        return speech_turn_original, speech_turn_merge