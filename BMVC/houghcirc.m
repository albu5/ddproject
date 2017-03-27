function [A, aoff, boff] = houghcirc(I, r)
boff = size(I,1)/2;
aoff = size(I,2)/2;
A = zeros(size(I));
for i = 1:size(I,1)
    i
    for j = 1:size(I,2)
        for theta = linspace(0, 2*pi, 30)
            a = j - r*cos(theta);
            b = i - r*sin(theta);
            try
                A(b+boff,a+aoff) = A(b+boff,a+aoff) + I(i,j);
            catch
            end
        end
    end
end

