function y = group_activity_feature(x)
N = size(x,1);

mean_and_hist = mean(x,1);
Theta = [0, 45, 90, 135, 180, -135, -90, -45, -180] * pi/180;

temps = [];
for i = 1:N
    for j = 1:N
        delx = x(i, 15) - x(j, 15);
        dely = x(i, 16) - x(j, 16);
        theta = atan2(dely, delx);
%         display(theta*180/pi), display(min(abs(Theta-theta))*180/pi)
        [~, thetaidx] = min(abs(Theta-theta));
        if thetaidx == 9
            thetaidx = 5;
        end
        if i ~= j
            temp = 1:8 == thetaidx;
        else
            temp = 1:8 == 0;
        end
        if numel(temps) == 0
            temps = temp;
        else
            temps = vertcat(temps, temp);
        end
    end
end

custom = mean(temps,1);
y = horzcat(mean_and_hist, custom);