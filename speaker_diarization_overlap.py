import torch
from pyannote.database import get_annotated
from pyannote.audio.pipeline.speaker_diarization import SpeakerDiarization
from pyannote.audio.pipeline.speech_turn_segmentation import SpeechTurnSegmentation, OracleSpeechTurnSegmentation
from pyannote.audio.pipeline.speech_activity_detection import SpeechActivityDetection
from pyannote.audio.pipeline.speaker_change_detection import SpeakerChangeDetection
from pyannote.audio.pipeline.speech_turn_clustering import SpeechTurnClustering
from pyannote.audio.pipeline.speech_turn_assignment import SpeechTurnClosestAssignment
from pyannote.pipeline import Pipeline
from sortedcontainers import SortedDict
from pyannote.pipeline.parameter import Uniform
from pyannote.core import Segment, Timeline, Annotation
from speech_turn_assignment_overlap import SpeechTurnClosestAssignmentMultiSpeaker, SpeechTurnClosestAssignmentMerge, SpeechTurnClosestAssignmentNew
from pathlib import Path
from typing import Optional
from typing import Union
from typing import Text

class SpeakerDiarizationOverlap(SpeakerDiarization):
    def __init__(
        self,
        mode,
        gnet, 
        device,
        sad_scores: Union[Text, Path] = None,
        scd_scores: Union[Text, Path] = None,
        embedding: Union[Text, Path] = None,
        metric: Optional[str] = "cosine",
        method: Optional[str] = "pool",
        evaluation_only: Optional[bool] = False,
        purity: Optional[float] = None,
    ):
        super(SpeakerDiarizationOverlap, self).__init__(sad_scores, scd_scores, embedding, metric, method, evaluation_only, purity)

        if mode == 'baseline':
            self.speech_turn_assignment = SpeechTurnClosestAssignment(
                embedding=self.embedding, metric=self.metric
            )
        elif mode == 'comp':
            self.speech_turn_assignment = SpeechTurnClosestAssignmentMultiSpeaker(
                gnet, device, embedding=self.embedding, metric=self.metric
            )
        elif mode == 'comp_ovl':
            self.speech_turn_assignment = SpeechTurnClosestAssignmentMerge(
                gnet, device, embedding=self.embedding, metric=self.metric
            )
        elif mode == 'baseline_ovl':
            self.speech_turn_assignment = SpeechTurnClosestAssignmentNew(
                embedding=self.embedding, metric=self.metric
            )

        self.dur = 1
        self.mode = mode
        
    def _turn_embeddings(self, current_file: dict):
        speech_turns = self.speech_turn_segmentation(current_file)
        embeddings = self.speech_turn_clustering._get_embeddings(current_file, speech_turns)
        return embeddings, speech_turns

    def _embeddings(self, current_file: dict):
        embeddings = self.speech_turn_clustering._embedding(current_file)
        return embeddings
    
    def __call__(self, current_file: dict) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Speaker diarization output.
        """

        # segmentation into speech turns
        speech_turns = self.speech_turn_segmentation(current_file)

        shorter_turns = Annotation(speech_turns.uri, speech_turns.modality)
        _tracks, _labels = [], set([])
        for key, value in speech_turns._tracks.items():
            start = key.start
            cnt = 0
            while True:
                if start + self.dur > key.end:
                    break
                _tracks.append((Segment(start, start + self.dur), {'_':value['_']+str(cnt)}))
                start += self.dur
                cnt += 1

            _tracks.append((Segment(start, key.end), dict(value)))

        shorter_turns._tracks = SortedDict(_tracks)
        for key, value in shorter_turns._tracks.items():
            _labels.update(value.values())

        shorter_turns._labels = {label: None for label in _labels}
        shorter_turns._labelNeedsUpdate = {label: True for label in _labels}

        shorter_turns._timeline = None
        shorter_turns._timelineNeedsUpdate = True

        
        if 'ovl' in self.mode:
            shorter_turns_intersect_ovl = Annotation(shorter_turns.uri, shorter_turns.modality)
            _tracks, _labels = [], set([])
            for key, value in shorter_turns._tracks.items():
                for cnt, ref_seg in enumerate(current_file['overlap']):
                    if ref_seg.intersects(key):
                        _tracks.append((Segment(max(key.start, ref_seg.start), min(key.end, ref_seg.end)), {'_':value['_']+str(cnt)}))
                        if key.start < ref_seg.start:
                            _tracks.append((Segment(key.start, ref_seg.start), {'_':value['_']+str(cnt)+str(cnt)}))
                        if key.end > ref_seg.end:
                            key = Segment(ref_seg.end, key.end)
                        else:
                            key = Segment(key.end, key.end)
                if key.end > key.start:
                    _tracks.append((key, dict(value)))

            shorter_turns_intersect_ovl._tracks = SortedDict(_tracks)
            for key, value in shorter_turns_intersect_ovl._tracks.items():
                _labels.update(value.values())

            shorter_turns_intersect_ovl._labels = {label: None for label in _labels}
            shorter_turns_intersect_ovl._labelNeedsUpdate = {label: True for label in _labels}

            shorter_turns_intersect_ovl._timeline = None
            shorter_turns_intersect_ovl._timelineNeedsUpdate = True
        # some files are only partially annotated and therefore one cannot
        # evaluate speaker diarization results on the whole file.
        # this option simply avoids trying to cluster those
        # (potentially messy) un-annotated refions by focusing only on
        # speech turns contained in the annotated regions.
        if self.evaluation_only:
            annotated = get_annotated(current_file)
            speech_turns = speech_turns.crop(annotated, mode="intersection")

        # in case there is one speech turn or less, there is no need to apply
        # any kind of clustering approach.
        if len(speech_turns) < 2:
            return speech_turns

        # split short/long speech turns. the idea is to first cluster long
        # speech turns (i.e. those for which we can trust embeddings) and then
        # assign each speech turn to the closest cluster.
        long_speech_turns = speech_turns.empty()
        shrt_speech_turns = speech_turns.empty()
        for segment, track, label in speech_turns.itertracks(yield_label=True):
            if segment.duration < self.min_duration:
                shrt_speech_turns[segment, track] = label
            else:
                long_speech_turns[segment, track] = label

        # in case there are no long speech turn to cluster, we return the
        # original speech turns (= shrt_speech_turns)
        if len(long_speech_turns) < 1:
            return speech_turns

        # first: cluster long speech turns
        long_speech_turns = self.speech_turn_clustering(current_file, long_speech_turns)

        # then: assign short speech turns to clusters
        long_speech_turns.rename_labels(generator="string", copy=False)

        if 'ovl' in self.mode:
            shorter_turns_intersect_ovl.rename_labels(generator="int", copy=False)
            speech_turns, speech_turns1 = self.speech_turn_assignment(
                current_file, shorter_turns_intersect_ovl, long_speech_turns
            )
            return speech_turns, speech_turns1
        elif self.mode == 'baseline':
            if len(shrt_speech_turns) > 0:
                shrt_speech_turns.rename_labels(generator="int", copy=False)
                shrt_speech_turns = self.speech_turn_assignment(
                    current_file, shrt_speech_turns, long_speech_turns
                )
            return long_speech_turns.update(shrt_speech_turns, copy=False).support(collar=0.0)
        elif self.mode == 'comp':
            shorter_turns.rename_labels(generator="int", copy=False)
            speech_turns = self.speech_turn_assignment(
                current_file, shorter_turns, long_speech_turns
            )
            return speech_turns

        # if self.ovl:
        #     if self.fg:
        #         shorter_turns_intersect_ovl.rename_labels(generator="int", copy=False)
        #         speech_turns, speech_turns1 = self.speech_turn_assignmentMerge(
        #             current_file, shorter_turns_intersect_ovl, long_speech_turns
        #         )
        #     else:
        #         shorter_turns_intersect_ovl.rename_labels(generator="int", copy=False)
        #         speech_turns, speech_turns1 = self.speech_turn_assignment_new(
        #             current_file, shorter_turns_intersect_ovl, long_speech_turns
        #         )
        # else:
        #     if self.fg:
        #         shorter_turns.rename_labels(generator="int", copy=False)
        #         speech_turns, speech_turns1 = self.speech_turn_assignmentMultispeaker(
        #             current_file, shorter_turns, long_speech_turns
        #         )
        #     else:
        #         if len(shrt_speech_turns) > 0:
        #             shrt_speech_turns.rename_labels(generator="int", copy=False)
        #             shrt_speech_turns = self.speech_turn_assignment(
        #                 current_file, shrt_speech_turns, long_speech_turns
        #             )
        #         return long_speech_turns.update(shrt_speech_turns, copy=False).support(collar=0.0), long_speech_turns.update(shrt_speech_turns, copy=False).support(collar=0.0)
            
        # return speech_turns, speech_turns1
