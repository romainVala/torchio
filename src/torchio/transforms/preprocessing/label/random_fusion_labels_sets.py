from typing import Sequence
from ....data.subject import Subject
from ...transform import TypeMaskingMethod
from .remap_labels import RemapLabels
from .label_transform import LabelTransform

import torch


class RandomFusionLabels(LabelTransform):
    r""" randomly Fusion a set of label to an other define set

    This transformation is not `invertible <invertibility>`_.

    Args:
        input_label_set: A sequence of label integers or string indexing an array (like for instance [4,'5:7', '14:')
        output_label_set: A sequence of 2 interger to give the 2 new labels values

        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    """

    def __init__(
        self,
        input_label_set: Sequence[int],
        output_label_set: Sequence[int],
        **kwargs,
    ):
        #remapping = {label: background_label for label in labels}
        #super().__init__(remapping, masking_method, **kwargs)
        super().__init__(**kwargs)
        self.input_label_set = input_label_set
        self.output_label_set = output_label_set

        self.args_names = ['input_label_set', 'output_label_set']

    def is_invertible(self):
        return False

    def apply_transform(self, subject):
        for name, image in self.get_images_dict(subject).items():
            if image.data.shape[0] > 1 : #4d label, either one hot, or Partial Volume
                original_label_values = [int(ii) for ii in range(0,image.data.shape[0])]
            else:
                original_label_values = image.data.int().unique().tolist() #adding casting to int because unique is not defind for half precision float
            
            # converting the input_label_set into a list of label values
            index_array = list(range(max(original_label_values)+1)) #there may be missing label values, this one is full
            mixing_index = []
            for ii in self.input_label_set:
                one_index = eval(f'index_array[{ii}]')
                if (not isinstance(one_index,list) ) : #case where you use int
                    one_index = [one_index]
                mixing_index += one_index
            #remove missing labels
            mixing_index.sort()
            for ii, val in reversed(list(enumerate(mixing_index))):
                if val not in original_label_values:
                    mixing_index.pop(ii)
            if len(mixing_index)==0:
                print(f'no label fusion for {image.path}')
                continue

            #random split into len(output_label_set) set            
            #shuffle
            mixing_index = torch.tensor(mixing_index)
            idx = torch.randperm(mixing_index.nelement())
            mixing_index = mixing_index.view(-1)[idx].view(mixing_index.size())
            
            nb_split = len(self.output_label_set) - 1
            idx = torch.randperm(len(mixing_index)-1) + 1 #so that no zero (counting from 1)
            index_cut = idx[:nb_split]
            index_cut = index_cut.tolist()
            index_cut = [0] + index_cut + [len(mixing_index)]

            remap_dic = {}
            for ii, out_value in enumerate(self.output_label_set):
                remap_dic.update({int(k):out_value for k in mixing_index[index_cut[ii]:index_cut[ii+1]]})
            #print(remap_dic)

            suj_with_one_label = Subject({'label': image})
            suj_remap = RemapLabels(remapping=remap_dic)(suj_with_one_label)
            image.set_data(suj_remap.label.data)

        return subject