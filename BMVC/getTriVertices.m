function [z1, z2] = getTriVertices(bb, pose, range)

margin = 45*pi/180;

Theta = [0, 45, 90, 135, 180, 225, 270, 315] * pi/180;

theta = Theta(pose);

x = bb(1);
y = bb(2);

sx = range * sqrt(bb(3)*bb(4));
sy = 1 * sqrt(bb(3)*bb(4));

a = x + sx * cos(theta+margin);
b = y + sy * sin(theta+margin);

p = x + sx * cos(theta-margin);
q = y + sy * sin(theta-margin);
z1 = [x; a; p];
z2 = [y; b; q];
