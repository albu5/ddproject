function dct2mot(stateInfo, outfile)
seq_det = -1*ones(sum(stateInfo.Xi(:)>0), 10);
deti = 1;
for frameid = stateInfo.frameNums
    aliveid = find(stateInfo.Xi(frameid, :) > 0);
    for ids = aliveid
        seq_det(deti,1) = frameid;
        seq_det(deti,2) = ids;
        seq_det(deti,3) = stateInfo.Xi(frameid, ids) - stateInfo.W(frameid, ids)/2;
        seq_det(deti,4) = stateInfo.Yi(frameid, ids) - stateInfo.H(frameid, ids);
        seq_det(deti,5) = stateInfo.W(frameid, ids);
        seq_det(deti,6) = stateInfo.H(frameid, ids);
        deti = deti + 1;
    end
end
fid = fopen(outfile, 'w');
fprintf(fid, '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\r\n', seq_det');
fclose(fid);


% dct2mot(stateInfo, 'home/ashish/Desktop/ped-det/data/eebuilding/track.txt')