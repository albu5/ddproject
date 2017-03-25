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