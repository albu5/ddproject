function nmfms_vot_test
write_dir = 'bag-results';
mkdir(write_dir);
image = sprintf('David/img/%04d.jpg',1);

params.bins = 16;
params.rdim = 2;
params.s = 0.3;
params.max_iter=50;
I = imread(image);
I = im2double(I);
I =I*255;
[~,J] = nmfreflect_model2(I,params);
imshow(J);
region = getrect;
display('press any key to continue ...')
pause
[state, params] = nmfms_initialize(imread(image), region);
idx = 1;
while true
    idx = idx + 1;
    write_frame=sprintf('%s//%d.jpeg',write_dir,idx);
    image=sprintf('David/img/%04d.jpg',idx);
    I = imread(image);
    % Perform a tracking step, obtain new region
    [state, region] = nmfms_update(state, params, region, I);
    
    h_x = (region(3)-1)/2;
    h_y = (region(4)-1)/2;
    center = [region(1)+h_x, region(2)+h_y];
    idx
    I = imread(image);
    I = im2double(I);
    I =I*255;
    [~,J] = nmfreflect_model2(I,params);
    I=draw_box(center(1),center(2),h_x+1,h_y+1,J);
    I=draw_box(center(1),center(2),h_x-1,h_y-1,I);
    I=draw_box(center(1),center(2),h_x,h_y,I);
    imshow(I)
    pause(0.0005)
    imwrite(I,write_frame);
    
end;
end

function [state, params] = nmfms_initialize(I, region, varargin)
params.bins = 16;
params.rdim = 2;
params.s = 0.3;
params.max_iter=50;
I = im2double(I);
I =I*255;
[~,J] = nmfreflect_model2(I,params);
J=J*255;
state.q_u1 = hist_model2(J,params.bins,region);
state.temph = state.q_u1;
state.bhat_coeff = bhattacharya_coeff(state.q_u1, state.temph, params.bins);
end

function [state, region] = nmfms_update(state, params, region, I, varargin)
I = im2double(I);
I =I*255;
[~,J] = nmfreflect_model2(I,params);
J=J*255;
[region,state.bhat_coeff,state.temph,~] = mean_shift2(J,state,region,params.bins);
state.q_u1=update_nmfmodel2(state);
end




% util functions ****************************************
function [Jill,J]=nmfreflect_model2(I,params)
rdim = params.rdim;
s = params.s;
max_iter= params.max_iter;
gamma = 1;

iptsetpref('ImshowBorder','tight');

[n, m, ~] =size(I);
[t_sizex,t_sizey,~]=size(I);
n = floor(n/t_sizex)*t_sizex;
m = floor(m/t_sizey)*t_sizey;

gray_ill = squeeze(sum(sum(I)));
gray_ill = gray_ill./sum(gray_ill);


logI = abs(-log(double(imresize(I+1,[n m])/255).^(1/gamma)));

if rdim == 2
    % t_size = img_size;
    V = [im2col(logI(:,:,1),[t_sizex t_sizey]); im2col(logI(:,:,2),[t_sizex t_sizey]); im2col(logI(:,:,3),[t_sizex t_sizey])];
    % V = [V,V];
else
    V = [im2col(logI(:,:,1),[t_sizex t_sizey],'distinct'); im2col(logI(:,:,2),[t_sizex y],'distinct'); im2col(logI(:,:,3),[t_sizex t_sizey],'distinct')];
end

sW = ones(1,rdim)*s; sW(1) = 0.001;

[W, H] = nmfsc_RGB(V, rdim, sW, [], 'temp',0, gray_ill, max_iter);

expW = exp(-W(:,1)*mean(H(1,:)));
Jill = reshape(expW,[t_sizex t_sizey 3]);
for j = 2:size(W,2)
    J = reshape(exp(-W(:,j)*mean(H(j,:))),[t_sizex t_sizey 3]);
    J = (J-min(J(:)))./(max(J(:))-min(J(:)));
    J = J./repmat(sum(J,3),[1 1 3]);
    J = J./max(J(:));
    %     figure(7); imshow(J);
end
end


function model = hist_model2(I,bins,region)
h_x = (region(3)-1)/2;
h_y = (region(4)-1)/2;
center = [region(1)+h_x, region(2)+h_y];
x1 = center(1)-h_x;
y1 = center(2)-h_y;
x2 = center(1)+h_x;
y2 = center(2)+h_y;

c=0;
binwidth=round(256/bins);
model= zeros(bins,bins,bins);
[sizex,sizey,~]=size(I);
for i=x1:1:x2,
    for j=y1:1:y2,
        tempx = (i-center(1))/h_x;
        tempy = (j-center(2))/h_y;
        
        dist =sqrt(tempx^2+tempy^2);
        
        if(dist>1)
            k=0;
        else
            k=1-dist;
        end
        
        if(j>0 && j<=sizex && i>0 && i<=sizey)
            
            r=floor(I(round(j),round(i),1)/binwidth)+1;
            g=floor(I(round(j),round(i),2)/binwidth)+1;
            b=floor(I(round(j),round(i),3)/binwidth)+1;
            
            model(r,g,b)=model(r,g,b)+k*k;
            c=c+k*k;
        end
    end
end
if( c~=0)
    model=model/c;
end
end

function rho = bhattacharya_coeff(q_u,p_u,bins)
%[sixx,sizyy]=size(q_u);
rho=0;

for i=1:1:bins
    for j=1:1:bins,
        for k=1:1:bins,
            rho=rho+sqrt(q_u(i,j,k)*p_u(i,j,k));
        end
    end
end
end


function [region,bhat_coeff,p_u,w] = mean_shift2(I,state,region,bins)
binwidth=round(256/bins);
% trial1=zeros(frames,20);
% trial2=zeros(frames,20);
q_u = state.q_u1;
for iter=1:1:20
    
    true=0;
    
    h_x = (region(3)-1)/2;
    h_y = (region(4)-1)/2;
    center = [region(1)+h_x, region(2)+h_y];
    x1 = center(1)-h_x;
    y1 = center(2)-h_y;
    x2 = center(1)+h_x;
    y2 = center(2)+h_y;
    
    p_u = hist_model(I,bins,center,h_x,h_y);
    
    [sizex,sizey,~] = size(I);
    
    oldcenter = center;
    
    sumt=0;
    sumx=0;
    sumy=0;
    
    for i=x1:1:x2,
        for j=y1:1:y2,
            
            tempx = (i-oldcenter(1))/h_x;
            tempy = (j-oldcenter(2))/h_y;
            
            if(j>0 && j<=sizex && i>0 && i<=sizey)
                
                true=true+1;
                r=floor(I(round(j),round(i),1)/binwidth)+1;
                g=floor(I(round(j),round(i),2)/binwidth)+1;
                b=floor(I(round(j),round(i),3)/binwidth)+1;
                
                
                if(p_u(r,g,b)==0.0)
                    weight=0;
                else
                    weight=sqrt(q_u(r,g,b)/p_u(r,g,b));
                    % temp4(i,j)=weight;
                end
            else
                weight=0;
            end
            sumt=sumt+weight;
            sumx=sumx+weight*tempx;
            sumy=sumy+weight*tempy;
        end
    end
    if(sumt ~= 0)
        sumx=sumx/sumt;
        sumy=sumy/sumt;
    end
    
    temx=oldcenter(1);
    temy=oldcenter(2);
    
    oldcenter(1)=oldcenter(1)+round(sumx*h_x);
    oldcenter(2)=oldcenter(2)+round(sumy*h_y);
    %  x_cor=oldcenter(1);
    %   trial1(frameint,iter)=x_cor;
    %   trial2(frameint,iter)=oldcenter(2);
    if((oldcenter(1)-temx)^2+(oldcenter(2)-temy)^2)<1
        break;
    else
    end
    center=round(oldcenter);
end

region = [oldcenter(1)-h_x, oldcenter(2)-h_y, 2*h_x+1, 2*h_y+1];
bhat_coeff= bhattacharya_coeff(q_u,p_u,bins);
w=weight;
end

function q_update = update_nmfmodel2(state)

if state.bhat_coeff < 0.9
    q_update = state.temph;
else
    q_update = state.q_u1;
end
end

function [W,H] = nmfsc_RGB( V, rdim, sW, sH, fname, showflag, est_ill, max_iter )
% nmfsc - non-negative matrix factorization with sparseness constraints
%
% SYNTAX:
% [W,H] = nmfsc( V, rdim, sW, sH, fname, showflag );f
%
% INPUTS:
% V          - data matrix
% rdim       - number of components (inner dimension of factorization)
% sW         - sparseness of W, in [0,1]. (give [] if no constraint)
% sH         - sparseness of H, in [0,1]. (give [] if no constraint)
% fname      - name of file to write results into
% showflag   - binary flag. if set then graphically show progress
%
% Note: Sparseness is measured on the scale [0,1] where 0 means
% completely distributed and 1 means ultimate sparseness.
%

% Check that we have non-negative data
if min(V(:))<0, error('Negative values in data!'); end

% Data dimensions
vdim = size(V,1);
samples = size(V,2);

% Create initial matrices
W = abs(randn(vdim,rdim));
%W = -(randn(vdim,rdim));
W(:,1) = cat(1, est_ill(1)*ones(vdim/3,1), est_ill(2)*ones(vdim/3,1), est_ill(3)*ones(vdim/3,1))*1;
W(:,1) = W(:,1)+randn(size(W(:,1)))/100;
H = abs(randn(rdim,samples));
H = H./(sqrt(sum(H.^2,2))*ones(1,samples));

% Make initial matrices have correct sparseness
if ~isempty(sW),
    L1a(1) = sqrt(vdim/3)-(sqrt(vdim/3)-1)*sW(1);
    W(:,1) = [projfunc(W(1:end/3,1),L1a(1),1,1); projfunc(W(end/3+1:end*2/3,1),L1a(1),1,1); projfunc(W(end*2/3+1:end,1),L1a(1),1,1)];
    for i=2:rdim,
        L1a(i) = sqrt(vdim)-(sqrt(vdim)-1)*sW(i);
        f = find(W(:,i)<0);
        W(:,i) = projfunc(abs(W(:,i)),L1a(i),1,1);
        %W(f,i) = -W(f,i);
    end
end
if ~isempty(sH),
    for i=1:rdim,
        L1s(i) = sqrt(samples)-(sqrt(samples)-1)*sH(i);
        H(i,:) = (projfunc(H(i,:)',L1s(i),1,1))';
    end
end

% Initialize displays
if showflag,
    figure(1); clf; % this will show the energies and sparsenesses
    figure(2); clf; % this will show the objective function
    drawnow;
end

% Calculate initial objective
objhistory = 0.5*sum(sum((V-W*H).^2));

% Initial stepsizes
stepsizeW = 1;
stepsizeH = 1;

timestarted = clock;

% Start iteration
iter = 0;
while iter < max_iter,
    
    % Show progress
    %fprintf('[%d]: %.5f\n',iter,objhistory(end));
    
    % Save every once in a while
    %     if rem(iter,5)==0,
    % 	elapsed = etime(clock,timestarted);
    % 	fprintf('Saving...');
    % 	save(fname,'W','H','sW','sH','iter','objhistory','elapsed');
    % 	fprintf('Done!\n');
    %     end
    
    % Show stats
    if showflag & (rem(iter,5)==0),
        figure(1);
        subplot(3,1,1); bar(sqrt(sum(W.^2)).*sqrt(sum(H'.^2)));
        cursW = (sqrt(vdim)-(sum(abs(W))./sqrt(sum(W.^2))))/(sqrt(vdim)-1);
        subplot(3,1,2); bar(cursW);
        cursH = (sqrt(samples)-(sum(abs(H'))./sqrt(sum(H'.^2)))) ...
            /(sqrt(samples)-1);
        subplot(3,1,3); bar(cursH);
        if iter>1,
            figure(2);
            plot(objhistory(2:end));
        end
        drawnow;
    end
    
    % Update iteration count
    iter = iter+1;
    
    % Save old values
    Wold = W;
    Hold = H;
    
    % ----- Update H ---------------------------------------
    
    if ~isempty(sH),
        
        % Gradient for H
        dH = W'*(W*H-V);
        begobj = objhistory(end);
        
        % Make sure we decrease the objective!
        while 1,
            
            % Take step in direction of negative gradient, and project
            Hnew = H - stepsizeH*dH;
            for i=1:rdim,
                Hnew(i,:) = (projfunc(Hnew(i,:)',L1s(i),1,1))';
            end
            
            % Calculate new objective
            newobj = 0.5*sum(sum((V-W*Hnew).^2));
            
            % If the objective decreased, we can continue...
            if newobj<=begobj,
                break;
            end
            
            % ...else decrease stepsize and try again
            stepsizeH = stepsizeH/2;
            fprintf('.');
            if stepsizeH<1e-200,
                fprintf('Algorithm converged.\n');
                return;
            end
            
        end
        
        % Slightly increase the stepsize
        stepsizeH = stepsizeH*1.2;
        H = Hnew;
        
    else
        
        % Update using standard NMF multiplicative update rule
        H = H.*(W'*V)./(W'*W*H + 1e-9);
        
        % Renormalize so rows of H have constant energy
        %H(1,:) = H(1,:)*0+mean(H(1,:));
        norms = sqrt(sum(H'.^2));
        
        %H = H./(norms'*ones(1,samples));
        
        %W = W.*(ones(vdim,1)*norms);
        
    end
    
    
    % ----- Update W ---------------------------------------
    
    if ~isempty(sW),
        
        % Gradient for W
        dW = (W*H-V)*H';
        begobj = 0.5*sum(sum((V-W*H).^2));
        
        % Make sure we decrease the objective!
        while 1,
            
            % Take step in direction of negative gradient, and project
            Wnew = W - stepsizeW*dW;
            norms = sqrt(sum(Wnew.^2));
            
            norms_r = sqrt(sum(Wnew(1:end/3,:).^2));
            norms_g = sqrt(sum(Wnew(end/3+1:end*2/3,:).^2));
            norms_b = sqrt(sum(Wnew(end*2/3+1:end,:).^2));
            Wnew(:,1)= [projfunc(Wnew(1:end/3,1),L1a(1)*norms_r(1),(norms_r(1)^2),1); ...
                projfunc(Wnew(end/3+1:end*2/3,1),L1a(1)*norms_g(1),(norms_g(1)^2),1); ...
                projfunc(Wnew(end*2/3+1:end,1),L1a(1)*norms_b(1),(norms_b(1)^2),1)];
            for i=2:rdim,
                f = find(Wnew(:,i)<0);
                Wnew(:,i) = projfunc(abs(Wnew(:,i)),L1a(i)*norms(i),(norms(i)^2),1);
                %Wnew(f,i) = -Wnew(f,i);
            end
            
            % Calculate new objective
            newobj = 0.5*sum(sum((V-Wnew*H).^2));
            
            % If the objective decreased, we can continue...
            if newobj<=begobj,
                break;
            end
            
            % ...else decrease stepsize and try again
            stepsizeW = stepsizeW/2;
            %   fprintf(',');
            if stepsizeW<1e-200,
                fprintf('Algorithm converged.\n');
                return;
            end
            
        end
        
        % Slightly increase the stepsize
        stepsizeW = stepsizeW*1.2;
        W = Wnew;
        
    else
        
        % Update using standard NMF multiplicative update rule
        W = W.*(V*H')./(W*H*H' + 1e-9);
        
    end
    
    
    % Calculate objective
    %     current_illu = W(:,1) * H(1,:);
    %     current_sur = W(:,2) * H(2,:);
    %     current_illu = min(current_illu,V);
    %     current_WH = current_illu + current_sur;
    %     newobj = 0.5*sum(sum((V-current_WH).^2));
    newobj = 0.5*sum(sum((V-W*H).^2));
    objhistory = [objhistory newobj];
    
    
end
end

function [v,usediters] = projfunc( s, k1, k2, nn )

% Solves the following problem:
% Given a vector s, find the vector v having sum(abs(v))=k1
% and sum(v.^2)=k2 which is closest to s in the euclidian sense.
% If the binary flag nn is set, the vector v is additionally
% restricted to being non-negative (v>=0).
%
% Written 2.7.2004 by Patrik O. Hoyer
%

% Problem dimension
N = length(s);

% If non-negativity flag not set, record signs and take abs
if ~nn,
    isneg = s<0;
    s = abs(s);
end

% Start by projecting the point to the sum constraint hyperplane
v = s + (k1-sum(s))/N;

% Initialize zerocoeff (initially, no elements are assumed zero)
zerocoeff = [];

j = 0;
while 1,
    
    % This does the proposed projection operator
    midpoint = ones(N,1)*k1/(N-length(zerocoeff));
    midpoint(zerocoeff) = 0;
    w = v-midpoint;
    a = sum(w.^2);
    b = 2*w'*v;
    c = sum(v.^2)-k2;
    alphap = (-b+sqrt(b^2-4*a*c))/(2*a);
    v = alphap*w + v;
    
    if all(v>=0),
        % We've found our solution
        usediters = j+1;
        break;
    end
    
    j = j+1;
    
    % Set negs to zero, subtract appropriate amount from rest
    zerocoeff = find(v<=0);
    v(zerocoeff) = 0;
    tempsum = sum(v);
    v = v + (k1-tempsum)/(N-length(zerocoeff));
    v(zerocoeff) = 0;
    
end

% If non-negativity flag not set, return signs to solution
if ~nn,
    v = (-2*isneg + 1).*v;
end

% Check for problems
if max(max(abs(imag(v))))>1e-10,
    error('Somehow got imaginary values!');
end
end
