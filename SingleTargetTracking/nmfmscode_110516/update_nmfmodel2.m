function q_update = update_nmfmodel2(state)

if state.bhat_coeff < 0.9
    q_update = state.temph;
else
    q_update = state.q_u1;    
end