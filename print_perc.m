function  print_perc(n,Nexp)
if (mod(10*n/Nexp,1) == 0) || n == 1
    if n/Nexp < 0.1
        perc = 0;
    elseif n/Nexp >= 0.1 && n/Nexp < 0.2
        perc = 10;
    elseif n/Nexp >= 0.2 && n/Nexp < 0.3
        perc = 20;
    elseif n/Nexp >= 0.3 && n/Nexp < 0.4
        perc = 30;
    elseif n/Nexp >= 0.4 && n/Nexp < 0.5
        perc = 40;
    elseif n/Nexp >= 0.5 && n/Nexp < 0.6
        perc = 50;
    elseif n/Nexp >= 0.6 && n/Nexp < 0.7
        perc = 60;
    elseif n/Nexp >= 0.7 && n/Nexp < 0.8
        perc = 70;
    elseif n/Nexp >= 0.8 && n/Nexp < 0.9
        perc = 80;
    elseif n/Nexp >= 0.9 && n/Nexp < 1
        perc = 90;
    else
        perc = 100;
    end
    
    if perc == 0
        fprintf ('0%%');
    elseif perc == 10
        fprintf('\b\b10%%');
    else
        fprintf('\b\b\b%d%%',perc);
        if perc == 100
            fprintf('\n');
        end
    end
end