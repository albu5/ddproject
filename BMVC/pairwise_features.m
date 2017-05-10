function phi = pairwise_features(desc1, desc2)

v1 = desc1(1:4);
v2 = desc2(1:4);
w1 = desc1(9);
h1 = desc1(10);
w2 = desc2(9);
h2 = desc2(10);
s1 = sqrt(w1 * h1);
s2 = sqrt(w2 * h2);
x1 = desc1(7:8);
x2 = desc2(7:8);
action1 = desc1(6);
action2 = desc2(6);

phi = [];
phi(end+1) = abs(v1(1) - v2(1));
phi(end+1) = abs(v1(2) - v2(2));
phi(end+1) = abs(v1(3) - v2(3));
phi(end+1) = abs(v1(4) - v2(4));
phi(end+1) = (abs(v1(1) - v2(1)) / sqrt(s1 * s2));
phi(end+1) = (abs(v1(2) - v2(2)) / sqrt(s1 * s2));
phi(end+1) = (abs(v1(3) - v2(3)) / sqrt(s1 * s2));
phi(end+1) = (abs(v1(4) - v2(4)) / sqrt(s1 * s2));
phi(end+1) = abs(w1 - w2);
phi(end+1) = abs(h1 - h2);
phi(end+1) = abs(x1(1) - x2(1));
phi(end+1) = abs(x1(2) - x2(2));

tx1 = desc1(11:13);
ty1 = desc1(14:16);
tx2 = desc2(11:13);
ty2 = desc2(14:16);

[ox, oy] = polybool('intersection', tx1, ty1, tx2, ty2);
pa = polyarea(ox, oy);
phi(end+1) = (pa);
phi(end+1) = (abs(action1-action2));