function q_update=update_nmfmodel2(temph,frameint,q_u1, bhat_coeff)

if bhat_coeff < 0.9
    q_update = temph;
    %  bhat_coeff3= bhattacharya_coeff(p_u,q_u1,bins);
else
    q_update=q_u1;
    %% if bhat_coeff3<0.85
    %           q_u1=temph(:,:,:,frameint-2);
    %           bhat_coeff4= bhattacharya_coeff(p_u,q_u1,bins);
    %          if bhat_coeff4<0.85
    %%       q_u1=q_u;
    %%  display('update2');
    %% end
    
end