import numpy as np
from scipy import io
from skimage.io import imread
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from numpy import genfromtxt


def gram_schmidt(vectors):
    basis = []
    dropped = False
    fatal = False
    for v in vectors:
        w = v - np.sum(np.dot(v, b) * b for b in basis)
        if (w > 1e-10).any():
            basis.append(w / np.linalg.norm(w))
        else:
            v = np.random.normal(size=v.shape)
            w = v - np.sum(np.dot(v, b) * b for b in basis)
            niter = 0
            while not (w > 1e-10).any():
                v = np.random.normal(size=v.shape)
                w = v - np.sum(np.dot(v, b) * b for b in basis)
                niter += 1
                if niter > 10:
                    fatal = True
                    break
            dropped = True
            basis.append(w / np.linalg.norm(w))
    return np.array(basis), dropped, fatal


def get_group_instance(gd):
    n_instances = gd.shape[0]
    choice = np.random.randint(0, n_instances-1)
    inst = gd[choice]
    persons = inst[0]
    features = []
    labels = []
    inst_dict = {}
    for person in persons:
        features.append(person['features'][0].tolist())
        labels.append(person['group'][0][0])
        inst_dict = {'features': np.array(features), 'labels': np.array(labels)}
    return inst_dict


def get_all_group_instances(gd):
    instances = []
    for inst in gd:
        persons = inst[0]
        features = []
        labels = []
        for person in persons:
            features.append(person['features'][0].tolist())
            labels.append(person['group'][0][0])
        inst_dict = {'features': np.array(features), 'labels': np.array(labels)}
        instances.append(inst_dict)


def read_cad_frames(data_dir, seqi, framei):
    seq_dir = data_dir + '/' + 'seq%2.2d' % seqi
    im_path = seq_dir + '/' + 'frame%4.4d' % framei + '.jpg'
    return imread(im_path)


def read_cad_annotations(anno_dir, seqi):
    anno_path = anno_dir + '/' + 'data_%2.2d' % seqi + '.txt'
    return genfromtxt(anno_path, delimiter=',')


def add_annotation(plt_axes, bbs, colour_str='r', line_width=1):
    rect = patches.Rectangle((bbs[0], bbs[1]), bbs[2], bbs[3], linewidth=line_width, edgecolor=colour_str,
                             facecolor='none')
    plt_axes.add_patch(rect)
    # return plt_axes


def get_interaction_features(annotation_data, frame_i, max_people=10):
    frame_i_idx = annotation_data[:, 0] == frame_i
    annotation_data_i = annotation_data[frame_i_idx, :]
    batch_x = []
    batch_y = []
    batch_pad = []
    for member in range(annotation_data_i.shape[0]):
        pair_feat = []
        pair_membership = []
        pair_pad = []
        for agent in range(annotation_data_i.shape[0]):
            if agent is not member:
                pair_feat.append(np.hstack((annotation_data_i[member, 10:], annotation_data_i[agent, 10:])).tolist())
                pair_membership.append(annotation_data_i[member, 8] == annotation_data_i[agent, 8])
                pair_pad.append(0)
        n_remaining = max_people - len(pair_membership) - 1
        for dummy in range(n_remaining):
            pair_feat.append(pair_feat[0])
            pair_membership.append(pair_membership[0])
            pair_pad.append(-2)

        batch_x.append(np.array(pair_feat).astype(np.float32))
        batch_y.append(np.expand_dims(np.array(pair_membership).astype(np.float32), axis=1))
        batch_pad.append(np.expand_dims(np.array(pair_pad).astype(np.float32), axis=1))

    return batch_x, batch_y, batch_pad


# group_data = io.loadmat('../group_data.mat')['trainX'][:, 0]
# data = get_group_instance(group_data)
# print(data)
# print(data['features'])
# print(data['labels'])
# print(data['features'][0])
#
'''
cad_dir = '../ActivityDataset'
annotations_dir = cad_dir + '/' + 'csvanno'
n = 1
f = 10
im = read_cad_frames(cad_dir, n, f)

anno_data = read_cad_annotations(annotations_dir, n)
print(anno_data.shape)

bx, by, bp = get_interaction_features(anno_data, 11)
print(bx[0].shape, by[0].shape, bp[0].shape)
print(len(bx))
print(bx[0])

fig, ax = plt.subplots(1)
ax.imshow(im)
add_annotation(ax, [0, 0, 50, 80], 'k', 2)
plt.show()
'''
