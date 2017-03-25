test.images = h5read('/tmp/crops/test.hdf5', '/crops');
test.labels = h5read('/tmp/crops/test.hdf5', '/labels');
test.pids = h5read('/tmp/crops/test.hdf5', '/pids');

valid.images = h5read('/tmp/crops/val.hdf5', '/crops');
valid.labels = h5read('/tmp/crops/val.hdf5', '/labels');
valid.pids = h5read('/tmp/crops/val.hdf5', '/pids');

train.images = h5read('/tmp/crops/train.hdf5', '/crops');
train.labels = h5read('/tmp/crops/train.hdf5', '/labels');
train.pids = h5read('/tmp/crops/train.hdf5', '/pids');

