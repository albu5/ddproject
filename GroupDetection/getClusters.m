function [clusters, adjM] = getClusters(scene)
adjM = getBinaryAdj(scene);
clusters = conncomp(graph(adjM));

